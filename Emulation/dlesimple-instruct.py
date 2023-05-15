#! /usr/bin/env python3 
 
from dlesimple import dle

import os 
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf  

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
 
dle = dle()

dle.config.network='cnn'
dle.config.dataset='ela_data'
dle.config.maptype='fKD'
dle.config.epochs = 100
dle.config.data_dir='fordle'
dle.config.results_dir='result'
dle.config.batch_size = 1
dle.config.learning_rate=0.001


#dle.config.clipnorm=0.0
#dle.config.nb_layers=0
#dle.config.nb_out_filter=16
#dle.config.conv_ker_size=5
  
dle.initialize()

dle.open_data()

dle.create_data_generator()
   
dle.train()

dle.predict()
