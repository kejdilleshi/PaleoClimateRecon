import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from pipeline import ModelBuilder, ModelEvaluator, DataLoader
from read_config_param import get_args
tf.random.set_seed(1234)

# Parse the command line arguments
args = get_args()
# Specify parameters
args.epochs = 5
args.data_dir = 'fordle'
args.dataset = 'new_topo'
args.batch_size=1
args.conv_ker_size=5
args.dropout_rate=0.1
args.test_topo = 'ETHI1.nc'
args.nb_blocks = 3
field_in = ['time', 'ela', 'topg']
field_out = ['gfp']


# Initialize model builder
model_builder = ModelBuilder(args)
model = model_builder.build_model(
            model_type='cnn', nb_inputs=3, nb_outputs=1)
model_builder.compile_model(model)
# Create an instance of the DataLoader class, specifying the path to the data directory
data_loader = DataLoader(args)
train_x, train_y, test_x, test_y = data_loader.split_data_3(
            field_in, field_out)
# Train model
history = model_builder.train(
    model, train_x, train_y, test_x, test_y)
# Evaluate the trained model
evaluator = ModelEvaluator(args, data_loader.set_scaling())
#evaluator.evaluate(model, test_x, test_y,10)
evaluator.plot_lc(history)

