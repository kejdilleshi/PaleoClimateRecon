#!/usr/bin/env python3

"""
dle.py contains all functions for the 2Dto2D-DLE (Deep Learning Emulator)
@author: Guillaume Jouvet
"""

import numpy as np
import os, sys, shutil, glob, math
import matplotlib.pyplot as plt
import datetime, time
import argparse
from   netCDF4 import Dataset
import tensorflow as tf
from scipy import interpolate
import xarray

def str2bool(v):
    return v.lower() in ("true", "1")

class dle:

    ####################################################################################
    #                                 INITIALIZATION
    ####################################################################################

    def __init__(self):
        """
            initialize the class DLE
        """

        self.parser = argparse.ArgumentParser(description="2Dto2D-DLE")

        self.read_config_param()

        self.config = self.parser.parse_args()
        
        self.set_pre_defined_mapping()

    def read_config_param(self):
        """
            read config file
        """

        self.parser.add_argument(
            "--dataset",
            type=str,
            default="pism_isoth_2000",
            help="Name of the dataset used for training",
        )
        self.parser.add_argument(
            "--maptype",
            type=str,
            default="f2",
            help="This use a predefined mapping, otherwise it is the folder name where to find fieldin.dat and fieldout.dat defining the mapping",
        )   
        self.parser.add_argument(
            "--network", 
            type=str, 
            default="cnn", 
            help="This is the type of network, it can be cnn or unet"
        )
        self.parser.add_argument(
            "--conv_ker_size", 
            type=int, 
            default=3, 
            help="Convolution kernel size for CNN"
        )
        self.parser.add_argument(
            "--nb_layers",
            type=int,
            default=16,
            help="Number of convolutional layers in the CNN",
        )
        self.parser.add_argument(
            "--nb_blocks",
            type=int,
            default=4,
            help="Number of block layer in the U-net",
        )
        self.parser.add_argument(
            "--nb_out_filter",
            type=int,
            default=32,
            help="Number of filters in the CNN or Unet",
        )
        self.parser.add_argument(
            "--activation",
            type=str,
            default="lrelu",
            help="Neural network activation function (lrelu = LeaklyRelu)",
        )
        self.parser.add_argument(
            "--dropout_rate",
            type=float,
            default=0,
            help="Neural network Drop out rate",
        )
        self.parser.add_argument(
            "--learning_rate", 
            type=float, 
            default=0.0001, 
            help="Learning rate"
        )
        self.parser.add_argument(
            "--clipnorm",
            type=float,
            default=0.5,
            help="Parameter that can clip the gradient norm (0.5)",
        )
        self.parser.add_argument(
            "--regularization",
            type=float,
            default=0.0,
            help="Regularization weight (0)",
        )
        self.parser.add_argument(
            "--batch_size", 
            type=int, 
            default=64, 
            help="Batch size for training (64)"
        )
        self.parser.add_argument(
            "--loss", 
            type=str, 
            default="mae", 
            help="Type of loss : mae or mse"
        )
        self.parser.add_argument(
            "--save_model_each",
            type=int,
            default=10000,
            help="The model is save each --save_model_each epochs",
        )
        self.parser.add_argument(
            "--data_augmentation", 
            type=str2bool, 
            default=True, 
            help="Augment data with some transformation"
        )
        self.parser.add_argument(
            "--epochs", 
            type=int, 
            default=100, 
            help="Number of epochs, i.e. pass over the data set at training"
        )
        self.parser.add_argument(
            "--seed", 
            type=int, 
            default=123, 
            help="Seed ID for reproductability"
        )
        self.parser.add_argument(
            "--data_dir",
            type=str,
            default="/home/gjouvet/DLE/data",
            help="Path of the data folder",
        )
        self.parser.add_argument(
            "--results_dir",
            type=str,
            default="/home/gjouvet/DLE/results",
            help="Path of the results folder",
        )
        self.parser.add_argument(
            "--verbose", 
            type=int, 
            default=1, 
            help="Verbosity level at training (1)"
        )
        self.parser.add_argument(
            "--usegpu",
            type=str2bool,
            default=True,
            help="use the GPU at training, this is nearly mandatory",
        )
        
    def set_pre_defined_mapping(self):

        self.mappings = {}
        
        self.mappings["fKD"] = {
            "fieldin": ["topg", "ela"],
            "fieldout": ["max_thk"],
        }
  
        self.naturalbounds = {}
        #
        self.naturalbounds["max_thk"]     = 2750.0
        self.naturalbounds["topg"]   	  = 4550.0
        self.naturalbounds["ela"]   	  = 1835.0
        #

    def initialize(self):
        """
            function initialize the absolute necessary
        """

        print(
            " -------------------- START 2Dto2D DLE --------------------------"
        )

        if self.config.network == "unet":
            self.network_depth = self.config.nb_blocks
        else:
            self.network_depth = self.config.nb_layers

        ndataset = self.config.dataset

        self.modelfile = os.path.join(self.config.results_dir, "model.h5")

        # creat fieldin, fieldout, fieldbound
               
        mapping = self.mappings[self.config.maptype]
        self.fieldin = mapping["fieldin"]
        self.fieldout = mapping["fieldout"]
        self.get_field_bounds() 
 
        # define the device to make computations (CPU or GPU)
        self.device_name = "/GPU:0" * self.config.usegpu + "/CPU:0" * (
            not self.config.usegpu
        )

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi

        if self.config.usegpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1" for multiple
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        print("-------------------- LIST of PARAMS --------------------------")
        for ck in self.config.__dict__:
            print("%30s : %s" % (ck, self.config.__dict__[ck]))

        print("----------------- CREATE PATH OF RESULTS ------------------------")
 
        if os.path.exists(self.config.results_dir):
            shutil.rmtree(self.config.results_dir)
            

        os.makedirs(self.config.results_dir, exist_ok=True)

        self.print_fields_in_out(self.config.results_dir)

        with open(os.path.join(self.config.results_dir, "dle-run-parameters.txt"), "w") as f:
            for ck in self.config.__dict__:
                print("%30s : %s" % (ck, self.config.__dict__[ck]), file=f)
                 
    def get_field_bounds(self):
        """
            get the fieldbounds (or scaling) from predfined values
        """

        self.fieldbounds = {}

        for f in self.fieldin:
            if f in self.naturalbounds.keys():
                self.fieldbounds[f] = self.naturalbounds[f]

        for f in self.fieldout:
            if f in self.naturalbounds.keys():
                self.fieldbounds[f] = self.naturalbounds[f]

    def read_fields_and_bounds(self, path):
        """
            get fields (input and outputs) from given file
        """

        self.fieldbounds = {}
        self.fieldin = []
        self.fieldout = []

        fid = open(os.path.join(path, "fieldin.dat"), "r")
        for fileline in fid:
            part = fileline.split()
            self.fieldin.append(part[0])
            self.fieldbounds[part[0]] = float(part[1])
        fid.close()

        fid = open(os.path.join(path, "fieldout.dat"), "r")
        for fileline in fid:
            part = fileline.split()
            self.fieldout.append(part[0])
            self.fieldbounds[part[0]] = float(part[1])
        fid.close()

    def print_fields_in_out(self, path):
        """
            print field inputs and outputs togther with field bounds (scaling)
        """

        fid = open(os.path.join(path, "fieldin.dat"), "w")
        for key in self.fieldin:
            fid.write("%s %.6f \n" % (key, self.fieldbounds[key]))
        fid.close()

        fid = open(os.path.join(path, "fieldout.dat"), "w")
        for key in self.fieldout:
            fid.write("%s %.6f \n" % (key, self.fieldbounds[key]))
        fid.close()

    ####################################################################################
    #                                 OPENING DATA
    #################################################################################### 

    def findsubdata(self, folder):
        """
            find the directory of 'test' and 'train' folder, to reference data
        """

        subdatasetpath = [f.path for f in os.scandir(folder) if f.is_dir()]

        subdatasetpath.sort(key=lambda x: (os.path.isdir(x), x))  # sort alphabtically

        subdatasetname = [f.split("/")[-1] for f in subdatasetpath]

        return subdatasetname, subdatasetpath

    def open_dataset(self, subdatasetpath):
        """
            open data assuming netcdf format
        """
 
        DATAIN = []
        DATAOUT = []

        for sf in subdatasetpath:
 
            nc = Dataset(os.path.join(sf, "ex.nc"), "r")

            allvar = {}
            for f in self.fieldin+self.fieldout:
                allvar[f] = np.array(nc.variables[f]).astype("float32")
              
            NPDATAIN = np.stack([allvar[f][:] for f in self.fieldin], axis=-1)
            NPDATAOUT = np.stack([allvar[f][:] for f in self.fieldout], axis=-1)
            
            nc.close()
 
            assert not np.any(np.isnan(NPDATAIN))
            assert not np.any(np.isnan(NPDATAOUT))

            DATAIN.append(NPDATAIN)
            DATAOUT.append(NPDATAOUT)
            
            del NPDATAIN,NPDATAOUT

        return DATAIN, DATAOUT
 
    def open_data(self):
        """
            Open data
        """

        print("----------------- OPEN  DATA        ------------------------")

        self.subdatasetname_train, self.subdatasetpath_train = self.findsubdata(
            os.path.join(self.config.data_dir, self.config.dataset, "train")
        )

        self.subdatasetname_test, self.subdatasetpath_test = self.findsubdata(
            os.path.join(self.config.data_dir, self.config.dataset, "test")
        )
 
        self.DATAIN_TRAIN, self.DATAOUT_TRAIN = self.open_dataset(  self.subdatasetpath_train )
  
        self.DATAIN_TEST, self.DATAOUT_TEST = self.open_dataset( self.subdatasetpath_test )

    ####################################################################################
    #                           PATCH AND DATA GENERARTOR
    ####################################################################################
    
    def datagenerator(self, DATAIN, DATAOUT):
        """
            Routine to serves to build a data generator
        """
  
        while True:  # Loop forever so the generator never terminates
        
            for (datain,dataout) in zip(DATAIN,DATAOUT):
                
                rec = datain.shape[0] // self.config.batch_size
 
                for k in range(0, rec):

                    ri = tf.constant([np.random.randint(0, 4), np.random.randint(0, 2), \
                                      np.random.randint(0, 2), np.random.randint(0, 2) ])
                     
                    X = tf.stack([datain[k::rec, :, :, f] /self.fieldbounds[self.fieldin[f]]  for f in range(len(self.fieldin))],axis=-1)
                    
                    Y = tf.stack([dataout[k::rec, :, :, f]/self.fieldbounds[self.fieldout[f]] for f in range(len(self.fieldout))],axis=-1)
                    
                    yield self.aug(X,ri), self.aug(Y,ri)
                    
                     # this is used to weight the loss function, can be usefull
#                    W = tf.stack([ 0.01*(datain[k::rec, :, :, 0]<=1.0)+(datain[k::rec, :, :, 0]>1.0) ],axis=-1)                            
#                    yield self.aug(X,ri), self.aug(Y,ri), self.aug(W,ri)
                
    def create_data_generator(self):
  
        self.trainSet = self.datagenerator( self.DATAIN_TRAIN, self.DATAOUT_TRAIN  )
        self.testSet = self.datagenerator(  self.DATAIN_TEST, self.DATAOUT_TEST )
        
        self.num_samples_train = np.sum([len(d) for d in self.DATAIN_TRAIN])
        self.num_samples_test  = np.sum([len(d) for d in self.DATAIN_TEST])
                    
    def aug(self, M, ri):
        for l in range(ri[0]):
            M = tf.image.rot90(M)
        if ri[1] == 1:
            M = tf.image.flip_left_right(M)
        if ri[2] == 1:
            M = tf.image.flip_up_down(M)
        if ri[3] == 1:
            M = tf.image.transpose(M)
        return M 
 
    #################################################################################
    ######                      DEFINE CNN OR UNET                        ###########
    ################################################################################# 

    def cnn(self, nb_inputs, nb_outputs):
        """
            Routine serve to build a convolutional neural network
        """

        inputs = tf.keras.layers.Input(shape=[None, None, nb_inputs])

        conv = inputs

        if self.config.activation == "lrelu":
            activation = tf.keras.layers.LeakyReLU(alpha=0.01)
        else:
            activation = tf.keras.layers.ReLU()

        for i in range(int(self.config.nb_layers)):

            conv = tf.keras.layers.Conv2D(
                filters=self.config.nb_out_filter,
                kernel_size=(self.config.conv_ker_size, self.config.conv_ker_size),
                kernel_regularizer=tf.keras.regularizers.l2(self.config.regularization),
                padding="same",
            )(conv)

            conv = activation(conv)

            conv = tf.keras.layers.Dropout(self.config.dropout_rate)(conv)

        outputs = conv

        outputs = tf.keras.layers.Conv2D(filters=nb_outputs, \
                                         kernel_size=(1, 1), \
                                         activation=None)(outputs)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs)

    def unet(self, nb_inputs, nb_outputs):
        """
            Routine serve to define a UNET network from keras_unet_collection
        """

        from keras_unet_collection import models

        layers = np.arange(int(self.config.nb_blocks))

        number_of_filters = [
            self.config.nb_out_filter * 2 ** (layers[i]) for i in range(len(layers))
        ]

        return models.unet_2d(
            (None, None, nb_inputs),
            number_of_filters,
            n_labels=nb_outputs,
            stack_num_down=2,
            stack_num_up=2,
            activation="LeakyReLU",
            output_activation=None,
            batch_norm=False,
            pool="max",
            unpool=False,
            name="unet",
        )
 
    #################################################################################
    ######                      TRAINING ROUTINE                          ###########
    ################################################################################# 

    def train(self):
        """
            THIS is the training routine
        """
 
        print("=========== Define the type of network (CNN or sth else) =========== ")

        nb_inputs = len(self.fieldin)
        nb_outputs = len(self.fieldout)

        self.model = getattr(self, self.config.network)(nb_inputs, nb_outputs)

        print("=========== Deinfe the optimizer =========== ")

        if self.config.clipnorm == 0.0:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.config.learning_rate
            )
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.config.learning_rate,
                clipnorm=self.config.clipnorm,
            )

        self.model.compile(
            loss=self.config.loss, optimizer=optimizer, metrics=["mae", "mse"]
        )  #  metrics=['mae','mse'] weighted_metrics=['mae','mse']

        step_per_epoch_train = self.num_samples_train // self.config.batch_size
        step_per_epoch_test = self.num_samples_test // self.config.batch_size

        print("=========== step_per_epoch : ", step_per_epoch_train)

        csv_logger = tf.keras.callbacks.CSVLogger(
            os.path.join(self.config.results_dir, "train-history.csv"),
            append=True,
            separator=" ",
        )

        model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.results_dir + "/model.{epoch:05d}.h5",
            save_freq="epoch",
            period=self.config.save_model_each,
        )
         
        class TimeHistory(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.times = []
        
            def on_epoch_begin(self, epoch, logs={}):
                self.epoch_time_start = time.time()
        
            def on_epoch_end(self, epoch, logs={}):
                self.times.append(time.time() - self.epoch_time_start)

        time_cb = TimeHistory()

        TerminateOnNaN_cb = tf.keras.callbacks.TerminateOnNaN()

        EarlyStopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=0,
            patience=100,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )

        cb = [
            csv_logger,
            time_cb,
            TerminateOnNaN_cb,
            EarlyStopping_cb,
            model_checkpoint_cb,
        ]

        print(self.model.summary())

        original_stdout = (
            sys.stdout
        )  # Save a reference to the original standard output
        with open(os.path.join(self.config.results_dir, "model-info.txt"), "w") as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print(self.model.summary())
            sys.stdout = (
                original_stdout  # Reset the standard output to its original value
            )

        print("=========== TRAINING =========== ")
        history = self.model.fit(
            self.trainSet,
            validation_data=self.testSet,
            # validation_freq=10,
            validation_steps=step_per_epoch_test,
            batch_size=self.config.batch_size,
            steps_per_epoch=step_per_epoch_train,
            epochs=self.config.epochs,
            callbacks=cb,
            verbose=self.config.verbose,
        )

        print("=========== plot learning curves =========== ")
        self.plotlearningcurves(self.config.results_dir, history.history)

        print("=========== save information on computational times =========== ")
        with open(
            os.path.join(self.config.results_dir, "train-time_callback.txt"), "w"
        ) as ff:
            print(time_cb.times, file=ff)

        self.history = history.history
        self.model.save(self.modelfile) 
        
    def plotlearningcurves(self, pathofresults, hist):

        fig = plt.figure(figsize=(8, 6))

        ax = fig.add_subplot(1, 1, 1)
        ax.plot(hist["loss"], "-", label="Train loss")
        ax.plot(hist["val_loss"], "--", label="Validation loss")
        ax.set_xlabel("Epoch", size=15)
        ax.set_ylabel("Loss", size=15)
        ax.legend(fontsize=15)
        ax.tick_params(axis="both", which="major", labelsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(pathofresults, "Learning-curve.pdf"))
        plt.close("all")

    #################################################################################
    ######                      PREDICT / EVALUATION ROUTINES             ###########
    ################################################################################# 

    def predict(self):
        """
            THIS is the predict routine for validation after training
        """
 
        print("===========     PREDICTING STEP =========== ")

        for kk, predglacier in enumerate(self.subdatasetname_test):

            DATAIN_T = [self.DATAIN_TEST[kk]]
            DATAOUT_T = [self.DATAOUT_TEST[kk]]

            DATAOUT_P, eval_time = self.evaluate(DATAIN_T)

            pathofresults = os.path.join(self.config.results_dir, predglacier)

            os.makedirs(pathofresults, exist_ok=True)

            self.plotresu(pathofresults, predglacier, DATAOUT_T, DATAOUT_P)
 
    def evaluate(self, DATAIN):
        """
            THIS function evaluates the neural network 
        """

        PRED = []

        for datain in DATAIN:

            Nt = datain.shape[0]
            Ny = datain.shape[1]
            Nx = datain.shape[2]

            if self.config.network == "unet":
                multiple_window_size = 8  # maybe this 2**(nb_layers-1)
                NNy = multiple_window_size * math.ceil(Ny / multiple_window_size)
                NNx = multiple_window_size * math.ceil(Nx / multiple_window_size)
            else:
                NNy = Ny
                NNx = Nx

            PAD = [[0, NNy - Ny], [0, NNx - Nx]]

            PREDI = np.zeros((Nt, Ny, Nx, len(self.fieldout)))

            eval_time = []

            for k in range(Nt):

                start_time = time.time()

                X = np.expand_dims(
                    np.stack(
                        [
                            np.pad(datain[k, :, :, kk], PAD) / self.fieldbounds[f]
                            for kk, f in enumerate(self.fieldin)
                        ],
                        axis=-1,
                    ),
                    axis=0,
                )

                Y = self.model.predict_on_batch(X)

                PREDI[k, :, :, :] = np.stack(
                    [
                        Y[0, :Ny, :Nx, kk] * self.fieldbounds[f]
                        for kk, f in enumerate(self.fieldout)
                    ],
                    axis=-1,
                )

                eval_time.append(time.time() - start_time)

            PRED.append(PREDI)

        return PRED, np.mean(eval_time)

    ####################################################################################
    #                             PLOT RESULT FROM PREDICT
    ####################################################################################

    def plotresu(self, pathofresults, dataset, DATAOUT_T, DATAOUT_P):

        for l in range(len(DATAOUT_T)):
            step = max(int(len(DATAOUT_T[l]) / 10), 1)
            for k in range(0, len(DATAOUT_T[l]), step):
                print("Plotting snapshopt n° : ", k)
                self.plotresu_f(pathofresults, dataset, DATAOUT_P[l][k], DATAOUT_T[l][k], k)

    def plotresu_f(self, pathofresults, dataset, pred_outputs, true_outputs, idx):
        """
            This routine permtis to plot predicted output against validation output
        """
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib import cm

        ny = pred_outputs.shape[0]
        nx = pred_outputs.shape[1]
        my_dpi = 100

        fig, (ax2, ax1) = plt.subplots(
            1, 2, figsize=(2 * ny / my_dpi, nx / my_dpi), dpi=my_dpi
        )

        m1 = tf.keras.metrics.MeanAbsoluteError()
        mae = m1(pred_outputs, true_outputs)

        m2 = tf.keras.metrics.RootMeanSquaredError()
        mse = m2(pred_outputs, true_outputs)

        pred_outputs0 = np.linalg.norm(pred_outputs, axis=2)
        true_outputs0 = np.linalg.norm(true_outputs, axis=2)

        valmaxx = max(
            [np.quantile(pred_outputs0, 0.999), np.quantile(true_outputs0, 0.999)]
        )

        ############################################

        fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(20, 10))

        ax1.set_title("PREDICTED, mae: %.5f, mse: %.5f" % (mae, mse))
        im1 = ax1.imshow(
            pred_outputs0,
            origin="lower",
            vmin=0,
            vmax=valmaxx,
            cmap=cm.get_cmap("viridis", 10),
        )
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar1 = plt.colorbar(im1, format="%.2f", cax=cax1)
        ax1.axis("off")

        ax2.set_title("TRUE, Id: %s" % str(idx))
        im2 = ax2.imshow(
            true_outputs0,
            origin="lower",
            vmin=0,
            vmax=valmaxx,
            cmap=cm.get_cmap("viridis", 10),
        )
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar2 = plt.colorbar(im2, format="%.2f", cax=cax2)
        ax2.axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                pathofresults, "predict-" + dataset + "_" + str(idx).zfill(4) + ".png"
            ),
            pad_inches=0,
            dpi=100,
        )
        plt.close("all")

        ############################################

        tod = pred_outputs0 - true_outputs0

        tod[np.abs(tod) < 0.03] = np.nan

        valmaxx2 = 25  # np.quantile(tod,0.99)

        fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))

        ax1.set_title("PREDICTED - TRUE")
        im1 = ax1.imshow(
            tod,
            origin="lower",
            vmin=-valmaxx2,
            vmax=valmaxx2,
            cmap=cm.get_cmap("RdBu", 10),
        )
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar1 = plt.colorbar(im1, format="%.2f", cax=cax1)
        ax1.axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                pathofresults,
                "predict-diff_" + dataset + "_" + str(idx).zfill(4) + ".png",
            ),
            pad_inches=0,
            dpi=100,
        )
        plt.close("all")

####################################################################################
#                               END
####################################################################################

 
