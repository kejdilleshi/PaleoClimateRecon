import argparse
def str2bool(v):
    return v.lower() in ("true", "1")
def get_args():
    """
            read config file
    """
    # Create the parser
    parser = argparse.ArgumentParser(description='Deep learning emulator pipeline')

    # Add arguments for hyperparameters
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100, 
        help="Number of epochs, i.e. pass over the data set at training"
    )
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default='./logs', 
        help="Directory where I save the callbacks"
    )
    parser.add_argument(
        "--conv_ker_size", 
        type=int, 
        default=3, 
        help="Convolution kernel size for CNN"
    )
    parser.add_argument(
        "--nb_layers",
        type=int,
        default=16,
        help="Number of convolutional layers in the CNN",
    )
    parser.add_argument(
        "--nb_blocks",
        type=int,
        default=4,
        help="Number of block layer in the U-net",
    )
    parser.add_argument(
        "--nb_out_filter",
        type=int,
        default=32,
        help="Number of filters in the CNN or Unet",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="lrelu",
        help="Neural network activation function (lrelu = LeaklyRelu)",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0,
        help="Neural network Drop out rate",
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=0.0001, 
        help="Learning rate"
    )
    parser.add_argument(
        "--clipnorm",
        type=float,
        default=0.5,
        help="Parameter that can clip the gradient norm (0.5)",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=0.0,
        help="Regularization weight (0)",
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=64, 
        help="Batch size for training (64)"
    )
    parser.add_argument(
        "--loss", 
        type=str, 
        default="mae", 
        help="Type of loss : mae or mse"
    )
    parser.add_argument(
        "--test_topo", 
        type=str, 
        default="ALPS.nc", 
        help="Type of loss : mae or mse"
    )
    parser.add_argument(
        "--verbose", 
        type=int, 
        default=1, 
        help="Verbosity level at training (1)"
    )
    parser.add_argument(
        "--train_test_ratio", 
        type=float, 
        default=0.8, 
        help="train_test_rati0 "
    )
    # Add arguments for input/output file paths
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="AllTopo",
        help="Name of the dataset used for training",
    )
    parser.add_argument(
        "--maptype",
        type=str,
        default="f2",
        help="This use a predefined mapping, otherwise it is the folder name where to find fieldin.dat and fieldout.dat defining the mapping",
    )  
    parser.add_argument(
        "--data_dir",
        type=str,
        default="fordle",
        help="Path of the data folder",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Path of the results folder",
    )
    # Add other arguments 
     
    parser.add_argument(
        "--train",
        type=str2bool,
        default=True,
        help="Set this to True if you wish to train",
    )
    parser.add_argument(
        "--predict",
        type=str2bool,
        default=True,
        help="Set this to True if you wish to predict",
    )
    parser.add_argument(
        "--network", 
        type=str, 
        default="cnn", 
        help="This is the type of network, it can be cnn or unet"
    )    
    parser.add_argument(
        "--resample_data",
        type=float,
        default=1.0,
        help="Coarsen data by averaging to generate other resolution",
    )
    parser.add_argument(
        "--include_test_in_train",
        type=str2bool,
        default=False,
        help="Force including the test data in the training dataset",
    )
    parser.add_argument(
        "--save_model_each",
        type=int,
        default=10000,
        help="The model is save each --save_model_each epochs",
    )
    parser.add_argument(
        "--data_augmentation", 
        type=str2bool, 
        default=True, 
        help="Augment data with some transformation"
    )
    parser.add_argument(
        "--data_stepping", 
        type=int, 
        default=1,
        help="This serves to take only a small part of data with a given step, practical for quick test"
    )   
    parser.add_argument(
        "--seed", 
        type=int, 
        default=123, 
        help="Seed ID for reproductability"
    )
    parser.add_argument(
        "--usegpu",
        type=str2bool,
        default=True,
        help="use the GPU at training, this is nearly mandatory",
    )
    

    
    # Parse the arguments
    args = parser.parse_args()
    
    return args