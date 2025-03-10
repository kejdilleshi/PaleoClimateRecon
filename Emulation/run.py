import sys
sys.path.append('/home/klleshi/Documents/PaleoClimateReconstruction')
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from pipeline import ModelBuilder, ModelEvaluator, DataLoader
from Constants import *
from read_config_param import get_args
tf.random.set_seed(1234)

# Parse the command line arguments
args = get_args()
#Specify parameters 
args.epochs=100
args.clipnorm=1.0
args.model_dir='results/9994/best_model.h5'
args.data_dir='fordle'
args.dataset='alps_transfer_c'
args.test_topo='ALPS.nc'
args.nb_blocks=4
field_in = ['time', 'ela', 'topg','A','beta','c']
field_out = ['gfp']


# Define the hyper parameters.
# Do not forget to define them in the Constans.py file.

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
 hp.hparams_config(
   hparams=[HP_BATCH_SIZE,HP_CONV_KER_SIZE,HP_LearningRata,HP_OPTIMIZER,HP_Architecture,HP_DropoutRata,HP_NB_LAYERS,HP_NB_FILTERS],
   metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
 )

def resize_data(train_x, train_y, test_x, test_y):
    #resize input data to shape (batch_size, 512, 512, nb_channels)
    train_x = tf.image.resize(
        train_x,
        size=(512, 512),
        method=tf.image.ResizeMethod.BILINEAR # specify the interpolation method here
    )
    test_x = tf.image.resize(
        test_x,
        size=(512, 512),
        method=tf.image.ResizeMethod.BILINEAR # specify the interpolation method here
    )
    # resize target data to shape (batch_size, 512, 512, nb_channels)
    train_y = tf.image.resize(
        train_y,
        size=(512, 512),
        method=tf.image.ResizeMethod.BILINEAR # specify the interpolation method here
    )
    test_y = tf.image.resize(
        test_y,
        size=(512, 512),
        method=tf.image.ResizeMethod.BILINEAR # specify the interpolation method here
    )
    return train_x, train_y, test_x, test_y
  
# Create a summary writer for each run
def run (run_name,hparams):
    with tf.summary.create_file_writer('logs/hparam_tuning/' + run_name).as_default():
        # Log the hyperparameters
        hp.hparams(hparams)  # record the values used in this trial

        ## Initialize model builder
        model_builder = ModelBuilder(args)
    
        # Build and compile model
        model = model_builder.build_model(model_type=hparams[HP_Architecture], nb_inputs=6, nb_outputs=1,hparams=hparams,model_path=args.model_dir, unfrozen_layers=10)
        
        model_builder.compile_model(model,hparams)
    
       ## Load and preprocess data
      	# Create an instance of the DataLoader class, specifying the path to the data directory
        data_loader = DataLoader(args)        
        train_x, train_y, test_x, test_y = data_loader.split_data_3(field_in, field_out)         
        if hparams[HP_Architecture]=='unet':
           # Call resize_data function to resize data if necessary
           train_x, train_y, test_x, test_y = resize_data(train_x, train_y, test_x, test_y)
        # Train model 
        history = model_builder.train(model, train_x, train_y, test_x, test_y,hparams)        
        ## Evaluate the trained model
        evaluator = ModelEvaluator(args,data_loader.set_scaling())
        evaluator.evaluate(model, test_x, test_y,nb_test=15)
        evaluator.plot_lc(history)
        evaluator.save_hparams(run_name=run_name,hparams={h.name: hparam[h] for h in hparam})
        
        # Accuracy metric for hyperparameter tracking
        accuracy = model.evaluate(test_x, test_y)
        mean_accuracy = sum(accuracy) / len(accuracy)
        # Log the metrics
        tf.summary.scalar(METRIC_ACCURACY, mean_accuracy, step=1)
    return model
    
# One run :
hparam = {
         HP_BATCH_SIZE: 1,
         HP_CONV_KER_SIZE: 7,
         HP_DropoutRata: 0.0,
         HP_LearningRata: 0.0001,
         HP_OPTIMIZER: 'adam',
         HP_Architecture: 'transfer_learning',
         HP_NB_FILTERS:64,
         HP_NB_LAYERS:16
            }
model=run('run_name', hparam)
## Print a summary of the model's architecture
model.summary()

exit() 
session_num = 0

for batch_size in HP_BATCH_SIZE.domain.values:
  for conv_ker_size in HP_CONV_KER_SIZE.domain.values:
    for learning_rata in HP_LearningRata.domain.values:
      for optimizer in HP_OPTIMIZER.domain.values:
        for architecture in HP_Architecture.domain.values:
          if architecture == 'unet':
            hparam = {
                HP_BATCH_SIZE: batch_size,
                HP_LearningRata: learning_rata,
                HP_OPTIMIZER: optimizer,
                HP_Architecture: architecture
            }
            
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparam[h] for h in hparam})
            args.results_dir='results/results_%d'%session_num
            # run(run_name, hparam)
            session_num += 1
          elif architecture=='cnn':
            for dropout in (HP_DropoutRata.domain.min_value, HP_DropoutRata.domain.max_value):
              for nb_layers in HP_NB_LAYERS.domain.values:
                for nb_filters in HP_NB_FILTERS.domain.values:
                  hparam = {
                      HP_BATCH_SIZE: batch_size,
                      HP_CONV_KER_SIZE: conv_ker_size,
                      HP_LearningRata: learning_rata,
                      HP_OPTIMIZER: optimizer,
                      HP_Architecture: architecture,
                      HP_DropoutRata: dropout,
                      HP_NB_LAYERS:nb_layers,
                      HP_NB_FILTERS:nb_filters
                  }
                  run_name = "run-%d" % session_num
                  print('--- Starting trial: %s' % run_name)
                  print({h.name: hparam[h] for h in hparam})
                  args.results_dir='results/results_%d'%session_num
                  # run(run_name, hparam)
                  session_num += 1
