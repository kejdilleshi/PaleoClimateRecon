import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorboard.plugins.hparams import api as hp
from keras_unet_collection import models
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from Constants import *

    
class ModelBuilder:
    tf.random.set_seed(0)
    def __init__(self, config):
        self.config = config
    
    def build_model(self, model_type, nb_inputs, nb_outputs,hparams,model_path=None, unfrozen_layers=None):
        if model_type == 'transfer_learning':
            return self.transfer_learning(model_path, unfrozen_layers)
        elif model_type == 'cnn':
            return self.cnn(nb_inputs, nb_outputs,hparams=hparams)
        elif model_type == 'unet':
            return self.unet(nb_inputs, nb_outputs)
        
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    # def cnn(self, nb_inputs, nb_outputs,hparams):
    #     """
    #         Routine serve to build a convolutional neural network
    #     """

    #     inputs = tf.keras.layers.Input(shape=[None, None, nb_inputs])

    #     conv = inputs
        
    #     if self.config.activation == "lrelu":
    #         activation = tf.keras.layers.LeakyReLU(alpha=0.01)
    #     else:
    #         activation = tf.keras.layers.ReLU()

    #     for i in range(int(hparams[HP_NB_LAYERS])):

    #         conv = tf.keras.layers.Conv2D(
    #             filters=hparams[HP_NB_FILTERS],
    #             kernel_size=(hparams[HP_CONV_KER_SIZE], hparams[HP_CONV_KER_SIZE]),
    #             kernel_regularizer=tf.keras.regularizers.l2(self.config.regularization),
    #             padding="same",
    #         )(conv)
            
    #         conv = activation(conv)

    #         conv = tf.keras.layers.Dropout(hparams[HP_DropoutRata])(conv) # add dropout layer here

    #     outputs = conv
    #     # Fully connected layer
    #     outputs = tf.keras.layers.Conv2D(filters=nb_outputs, \
    #                                      kernel_size=(1, 1), \
    #                                      activation=None)(outputs)

    #     return tf.keras.models.Model(inputs=inputs, outputs=outputs)

    def cnn(self, nb_inputs, nb_outputs,hparams):
     """
         Routine serve to build a convolutional neural network
     """
    
     inputs = tf.keras.layers.Input(shape=[None, None, nb_inputs])
    
     conv = inputs
    
     if self.config.activation == "lrelu":
         activation = tf.keras.layers.LeakyReLU(alpha=0.01)
     else:
         activation = tf.keras.layers.ReLU()
    
     skip_connections = []
     for i in range(int(hparams[HP_NB_LAYERS])):
        
         conv = tf.keras.layers.Conv2D(
             filters=hparams[HP_NB_FILTERS],
             kernel_size=(hparams[HP_CONV_KER_SIZE], hparams[HP_CONV_KER_SIZE]),
             kernel_regularizer=tf.keras.regularizers.l2(self.config.regularization),
             padding="same",
         )(conv)
    
         conv = activation(conv)
    
         conv = tf.keras.layers.Dropout(hparams[HP_DropoutRata])(conv) # add dropout layer here
    
         if i % 2 == 0:  # Add skip connection every 2 layers
             skip_connections.append(conv)
    
         if i >= 2 and (i - 2) % 2 == 0:  # Add the skip connection to the current layer
             conv = tf.keras.layers.Add()([conv, skip_connections[-1]])
    
     outputs = conv
     # Fully connected layer
     outputs = tf.keras.layers.Conv2D(filters=nb_outputs, \
                                      kernel_size=(1, 1), \
                                      activation=None)(outputs)
    
     return tf.keras.models.Model(inputs=inputs, outputs=outputs)

            
    def unet(self, nb_inputs, nb_outputs):
        """
            Routine serve to define a UNET network from keras_unet_collection
        """

        

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
    def compile_model(self, model, hparams):
        if hparams[HP_OPTIMIZER] == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=hparams[HP_LearningRata],clipnorm=self.config.clipnorm)
            loss = tf.keras.losses.MeanAbsoluteError()
            metrics = [tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()]
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            
        elif hparams[HP_OPTIMIZER] == 'lbfgs':
            import tensorflow_probability as tfp
            initial_position=np.arange(1, 0, -1, dtype='float64')
            optimizer = tfp.optimizer.lbfgs_minimize(
                loss,
                initial_position=initial_position,
                tolerance=1e-8,
                max_iterations=100
            )
            loss = tf.keras.losses.MeanAbsoluteError()
            metrics = [tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()]
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    
    
    def train(self, model, train_x, train_y, test_x, test_y,hparams,Data_augment=False):
        """
        Train the model on the given data.
    
        :param model: The compiled model to train.
        :param train_x: The training data features.
        :param train_y: The training data labels.
        :param test_x: The test data features.
        :param test_y: The test data labels.
        """
        # Create an instance of the ImageDataGenerator class
        datagen = ImageDataGenerator(
            rotation_range=20, # rotate the image up to 20 degrees
            horizontal_flip=True, # randomly flip the image horizontally
            vertical_flip=True # randomly flip the image vertically
            )
        # Fit the generator on your training data
        #datagen.fit(train_x)
    
        # Create a ModelCheckpoint callback to save only the best model
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.results_dir + '/best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        early_stopping = EarlyStopping(
        monitor='loss', # the quantity to monitor
        patience=20, # number of epochs with no improvement after which training will be stopped
        restore_best_weights=True # whether to restore model weights from the epoch with the best value of the monitored quantity
        )
        
        if Data_augment==True:
        
            history = model.fit(
                datagen.flow(train_x, train_y, batch_size=hparams[HP_BATCH_SIZE]),
                epochs=self.config.epochs,
                validation_data=(test_x, test_y),
                callbacks=[tf.keras.callbacks.TensorBoard(log_dir=self.config.log_dir,histogram_freq=1,update_freq='batch'),checkpoint,hp.KerasCallback(self.config.log_dir, hparams),early_stopping]
                )
        else:
            history = model.fit(
                train_x,
                train_y,
                batch_size=hparams[HP_BATCH_SIZE],
                epochs=self.config.epochs,
                validation_data=(test_x, test_y),
                callbacks=[tf.keras.callbacks.TensorBoard(log_dir=self.config.log_dir,histogram_freq=1,update_freq='batch'),checkpoint,hp.KerasCallback(self.config.log_dir, hparams),early_stopping]
            )   
    
        return history
    def transfer_learning(self, model_path, unfrozen_layers):
        """
        Load a pre-trained model and freeze the layers except for the last 'unfrozen_layers' layers

        Parameters:
        model_path (str): The path to the pre-trained model
        unfrozen_layers (int): The number of layers to leave unfrozen

        Returns:
        model: a Model instance with the specified layers unfrozen
        """
        # Load the pre-trained model
        model = tf.keras.models.load_model(model_path)

        # Freeze all the layers before the 'unfrozen_layers' layers
        for layer in model.layers[:-unfrozen_layers]:
            layer.trainable = False

        return model