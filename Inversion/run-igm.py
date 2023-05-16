#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from igm import Igm

glacier = Igm()

glacier.config.working_dir = ''
#glacier.config.iceflow_model_lib_path= '../../model-lib/f12_cfsflow'
glacier.config.iceflow_model_lib_path = '../Emulation/result'

glacier.config.plot_result = False
glacier.config.plot_live = False
glacier.config.observation_file = 'obs_test.nc'  # this is the main input file

glacier.config.opti_output_freq     = 100     # Frequency for output
glacier.config.opti_nbitmax         = 10000   # Number of it. for the optimization
# weight for the regul. of strflowctrl
glacier.config.opti_regu_param_strflowctrl = 1.0

# test
ones=np.ones(glacier.config.opti_nbitmax)
ones[:int(glacier.config.opti_nbitmax/3)]=1000
ones[int(glacier.config.opti_nbitmax/3):int(2*glacier.config.opti_nbitmax/3)]=500
ones[int(2*glacier.config.opti_nbitmax/3):int(glacier.config.opti_nbitmax)]=300
##


glacier.config.opti_step_size = 0.001
glacier.config.opti_regu_grad_ela=ones
glacier.config.opti_smooth_anisotropy_factor = 0.2   # Smooth anisotropy factor

# Uncomment for Optimization scheme O
glacier.config.opti_control = ['ela', 'topg']
glacier.config.opti_cost = ['ela', 'topg']



glacier.initialize()
with tf.device(glacier.device_name):
    glacier.load_ncdf_data(glacier.config.observation_file)
    glacier.initialize_fields()
    glacier.optimize2()
