import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorboard.plugins.hparams import api as hp
import numpy as np
tf.random.set_seed(1234)
class ModelBuilder:
   def __init__(self, config):
       self.config = config

   def build_model(self, model_type, nb_inputs, nb_outputs):
       if model_type == 'cnn':
           return self.cnn(nb_inputs, nb_outputs)
       elif model_type == 'unet':
           return self.unet(nb_inputs, nb_outputs)
       else:
           raise ValueError(f"Invalid model type: {model_type}")

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
   def compile_model(self, model):
       optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
       loss = tf.keras.losses.MeanAbsoluteError()
       metrics = [tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()]
       model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
       
   
   
   def train(self, model, train_x, train_y, test_x, test_y,Data_augment=False):
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
       datagen.fit(train_x)
   
       # Create a ModelCheckpoint callback
       checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.results_dir + '/best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1)
      
       if Data_augment==True:
       
           history = model.fit(
               datagen.flow(train_x, train_y, batch_size=self.config.batch_size),
               epochs=self.config.epochs,
               validation_data=(test_x, test_y),
               callbacks=[tf.keras.callbacks.TensorBoard(log_dir=self.config.log_dir,histogram_freq=1,update_freq='batch'),checkpoint,]
               )
       else:
           history = model.fit(
               train_x,
               train_y,
               batch_size=self.config.batch_size,
               epochs=self.config.epochs,
               validation_data=(test_x, test_y),
               callbacks=[tf.keras.callbacks.TensorBoard(log_dir=self.config.log_dir,histogram_freq=1,update_freq='batch'),checkpoint]
           )   
   
       return history

# class ModelBuilder:
#     def __init__(self, config):
#         self.config = config
    
#     def build_model(self, model_type, nb_inputs, nb_outputs,hparams):
#         if model_type == 'cnn':
#             return self.cnn(nb_inputs, nb_outputs,hparams=hparams)
#         elif model_type == 'unet':
#             return self.unet(nb_inputs, nb_outputs)
#         else:
#             raise ValueError(f"Invalid model type: {model_type}")

#     def cnn(self, nb_inputs, nb_outputs,hparams):
#         """
#             Routine serve to build a convolutional neural network
#         """

#         inputs = tf.keras.layers.Input(shape=[None, None, nb_inputs])

#         conv = inputs
        
#         if self.config.activation == "lrelu":
#             activation = tf.keras.layers.LeakyReLU(alpha=0.01)
#         else:
#             activation = tf.keras.layers.ReLU()

#         for i in range(int(hparams[HP_NB_LAYERS])):

#             conv = tf.keras.layers.Conv2D(
#                 filters=hparams[HP_NB_FILTERS],
#                 kernel_size=(hparams[HP_CONV_KER_SIZE], hparams[HP_CONV_KER_SIZE]),
#                 kernel_regularizer=tf.keras.regularizers.l2(self.config.regularization),
#                 padding="same",
#                 # kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.5)
#             )(conv)
            
#             conv = activation(conv)

#             conv = tf.keras.layers.Dropout(hparams[HP_DropoutRata])(conv)

#         outputs = conv

#         outputs = tf.keras.layers.Conv2D(filters=nb_outputs, \
#                                          kernel_size=(1, 1), \
#                                          activation=None)(outputs)

#         return tf.keras.models.Model(inputs=inputs, outputs=outputs)

#     def unet(self, nb_inputs, nb_outputs):
#         """
#             Routine serve to define a UNET network from keras_unet_collection
#         """

#         from keras_unet_collection import models

#         layers = np.arange(int(self.config.nb_blocks))

#         number_of_filters = [
#             self.config.nb_out_filter * 2 ** (layers[i]) for i in range(len(layers))
#         ]

#         return models.unet_2d(
#             (None, None, nb_inputs),
#             number_of_filters,
#             n_labels=nb_outputs,
#             stack_num_down=2,
#             stack_num_up=2,
#             activation="LeakyReLU",
#             output_activation=None,
#             batch_norm=False,
#             pool="max",
#             unpool=False,
#             name="unet",
#         )         
#     def compile_model(self, model, hparams):
#         if hparams[HP_OPTIMIZER] == 'adam':
#             optimizer = tf.keras.optimizers.Adam(learning_rate=hparams[HP_LearningRata])
#             loss = tf.keras.losses.MeanAbsoluteError()
#             metrics = [tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()]
#             model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
#         elif hparams[HP_OPTIMIZER] == 'lbfgs':
#             import tensorflow_probability as tfp
#             initial_position=np.arange(1, 0, -1, dtype='float64')
#             optimizer = tfp.optimizer.lbfgs_minimize(
#                 loss,
#                 initial_position=initial_position,
#                 tolerance=1e-8,
#                 max_iterations=100
#             )
#             loss = tf.keras.losses.MeanAbsoluteError()
#             metrics = [tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()]
#             model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    
    
#     def train(self, model, train_x, train_y, test_x, test_y,hparams,Data_augment=False):
#         """
#         Train the model on the given data.
    
#         :param model: The compiled model to train.
#         :param train_x: The training data features.
#         :param train_y: The training data labels.
#         :param test_x: The test data features.
#         :param test_y: The test data labels.
#         """
#         # Create an instance of the ImageDataGenerator class
#         datagen = ImageDataGenerator(
#             rotation_range=20, # rotate the image up to 20 degrees
#             horizontal_flip=True, # randomly flip the image horizontally
#             vertical_flip=True # randomly flip the image vertically
#             )
#         # Fit the generator on your training data
#         datagen.fit(train_x)
    
#         # Create a ModelCheckpoint callback to save only the best model
#         checkpoint = tf.keras.callbacks.ModelCheckpoint(
#             filepath=self.config.results_dir + '/best_model.h5',
#             monitor='val_loss',
#             save_best_only=True,
#             mode='min',
#             verbose=1
#         )
#         if Data_augment==True:
        
#             history = model.fit(
#                 datagen.flow(train_x, train_y, batch_size=hparams[HP_BATCH_SIZE]),
#                 epochs=self.config.epochs,
#                 validation_data=(test_x, test_y),
#                 callbacks=[tf.keras.callbacks.TensorBoard(log_dir=self.config.log_dir,histogram_freq=1,update_freq='batch'),checkpoint,hp.KerasCallback(self.config.log_dir, hparams)]
#                 )
#         else:
#             history = model.fit(
#                 train_x,
#                 train_y,
#                 batch_size=hparams[HP_BATCH_SIZE],
#                 epochs=self.config.epochs,
#                 validation_data=(test_x, test_y),
#                 callbacks=[tf.keras.callbacks.TensorBoard(log_dir=self.config.log_dir,histogram_freq=1,update_freq='batch'),checkpoint,hp.KerasCallback(self.config.log_dir, hparams)]
#             )   
    
#         return history

        
