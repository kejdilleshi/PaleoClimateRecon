import sys
sys.path.append('/home/klleshi/Documents/PaleoClimateReconstruction')
from pipeline import Optimizer
from read_config_param import get_args


# Parse the command line arguments
args = get_args()
#Specify parameters 
args.data_dir='Data'
args.dataset='alps_topo'

args.modeldir='../models/results7/model.h5'
args.save_freq =2000
args.opti_nbitmax=20000
# Define lists of values for the arguments
learning_rates = [0.0001]
opti_regu_grad_ela_values = [500000000]
opti_regu_grad_time_values = [0]
l1_values=[0]
l2_values=[1.0]
l_files=['obs_t3.nc','obs_t4.nc','obs_t5.nc','obs_bini_t3.nc','obs_bini_t4.nc','obs_bini_t5.nc']

session_num=100



def run ():
    # excecute the optimization
    optimizer = Optimizer(args)
    optimizer.optimize()
    

# #asign new values
# args.opti_regu_grad_time=500000000
# args.opti_regu_grad_ela=500000000
# args.learning_rate=0.0001
# args.l1_lambda=0
# args.l2_lambda=1.0
# args.obs_file='observation.nc'
# args.results_dir='results/100'
# # Run the command
# run ()
# exit()


# Iterate through the combinations of values

for file in l_files:
    for learning_rate in learning_rates:
        for opti_regu_grad_ela in opti_regu_grad_ela_values:
            for opti_regu_grad_time in opti_regu_grad_time_values:
                for l1 in l1_values:
                    for l2 in l2_values:
                    
                        #asign new values
                        args.opti_regu_grad_time=opti_regu_grad_time
                        args.opti_regu_grad_ela=opti_regu_grad_ela
                        args.learning_rate=learning_rate
                        args.l1_lambda=l1
                        args.l2_lambda=l2
                        args.obs_file=file  
                        args.results_dir='results/%d'%session_num
                        # Run the command
                        run()
                        print(session_num, file, opti_regu_grad_time, opti_regu_grad_ela,learning_rate,l1,l2)
                        session_num+=1
