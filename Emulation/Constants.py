from tensorboard.plugins.hparams import api as hp
    
# Hyper parameters  
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([1,10]))
HP_CONV_KER_SIZE = hp.HParam('conv_ker_size', hp.Discrete([5,7]))
HP_LearningRata = hp.HParam('learning_rata', hp.Discrete([0.001,0.0001]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))
HP_Architecture= hp.HParam('architecture', hp.Discrete(['cnn', 'unet']))
HP_DropoutRata = hp.HParam('dropout', hp.RealInterval(0.0, 0.1))
HP_NB_LAYERS = hp.HParam('nb_layers', hp.Discrete([16,32]))
HP_NB_FILTERS = hp.HParam('nb_filters', hp.Discrete([32,64]))

METRIC_ACCURACY = 'accuracy'

