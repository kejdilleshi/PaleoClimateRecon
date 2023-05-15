#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from igm import Igm

glacier = Igm()

glacier.config.working_dir = ''
#glacier.config.iceflow_model_lib_path= '../../model-lib/f12_cfsflow'
glacier.config.iceflow_model_lib_path = '/home/klleshi/Desktop/igm/model-lib/fKD_IGM_KLL_22_a'

glacier.config.plot_result = False
glacier.config.plot_live = False
glacier.config.observation_file = 'observation.nc'  # this is the main input file

glacier.config.opti_output_freq     = 100     # Frequency for output
glacier.config.opti_nbitmax         = 10000   # Number of it. for the optimization
glacier.config.opti_regu_param_thk  = 5.0   # weight for the regul. of thk
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

glacier.config.opti_usurfobs_std = 5.0   # Tol to fit top ice surface
glacier.config.opti_velsurfobs_std = 3.0   # Tol to fit surface speeds
glacier.config.opti_thkobs_std = 5.0   # Tol to fit ice thk profiles
glacier.config.opti_divfluxobs_std = 1.0   # Tol to fit top ice surface

# Uncomment for Optimization scheme O
glacier.config.opti_control = ['ela', 'topg']
glacier.config.opti_cost = ['ela', 'topg']

# Uncomment for Optimization scheme $O_{-\tilde{A}}$
# glacier.config.opti_control=['thk','usurf']
# glacier.config.opti_cost=['velsurf','thk','usurf','divfluxfcz','icemask']

# Uncomment for Optimization scheme $O_{-\tilde{A},h}$
# glacier.config.opti_control=['thk','usurf']
# glacier.config.opti_cost=['velsurf','usurf','divfluxfcz','icemask']

# Uncomment for Optimization scheme $O_{-d}$
# glacier.config.opti_control=['thk','strflowctrl','usurf']
# glacier.config.opti_cost=['velsurf','thk','usurf','icemask']

# Uncomment for Optimization scheme $O_{-s}$
# glacier.config.opti_control=['thk','strflowctrl']
# glacier.config.opti_cost=['velsurf','thk','divfluxfcz','icemask']

glacier.initialize()
with tf.device(glacier.device_name):
    glacier.load_ncdf_data(glacier.config.observation_file)
    glacier.initialize_fields()
    glacier.optimize2()
#with tf.device(glacier.device_name):
#    glacier.load_ncdf_data(glacier.config.observation_file)
#    glacier.initialize_fields()
#    glacier.optimize()
#
#glacier.print_all_comp_info()
