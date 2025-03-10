from pipeline import Optimizer
from read_config_param import get_args


# Parse the command line arguments
args = get_args()
# Specify parameters
args.data_dir = "fordle"
args.dataset = "alps_topo"

args.modeldir = "results/BrutTraining_AllData/best_model.h5"
args.save_freq = 1000
args.opti_nbitmax = 3000
field_in = ["time", "ela", "topg", "A", "beta", "c"]

# Define lists of values for the arguments
learning_rates = [0.0001]
opti_regu_grad_ela_values = [1000]
opti_regu_grad_time_values = [0]
l1_values = [0]
l2_values = [1.0]
l_files = [
    "obs_t5_A78b0.016c2.1.nc",
    "obs_t4_A78b0.008c2.1.nc",
    "obs_t3_A78b0.008c2.1.nc",
    "obs_t4_A39b0.008c2.1.nc",
    "obs_t3_A156b0.008c2.1.nc",
    "obs_t5_A39b0.008c2.1.nc",
    "obs_t5_A78b0.004c2.1.nc",
    "obs_t5_A78b0.008c2.1.nc",
    "obs_t4_A78b0.004c2.1.nc",
    "obs_t4_A78b0.016c2.1.nc",
    "obs_t5_A156b0.008c2.1.nc",
    "obs_t4_A156b0.008c2.1.nc",
    "obs_t3_A78b0.016c2.1.nc",
    "obs_t3_A78b0.004c2.1.nc",
    "obs_t3_A39b0.008c2.1.nc",
]
session_num = 0


def run():
    # excecute the optimization
    optimizer = Optimizer(args)
    # optimizer.optimize()
    optimizer.optimize_extent()


# asign new values
args.opti_regu_grad_time=0.0
args.opti_regu_grad_ela=1000
args.learning_rate=0.0001
args.l1_lambda=1.0
args.l2_lambda=0
args.obs_file="obs_t5_A156b0.008c2.1.nc"
args.results_dir='results/test_inversion_spatial3'
# Run the command
# run ()
# exit()


# Iterate through the combinations of values

for file in l_files:
    for learning_rate in learning_rates:
        for opti_regu_grad_ela in opti_regu_grad_ela_values:

            # asign new values
            args.opti_regu_grad_ela = opti_regu_grad_ela
            args.learning_rate = learning_rate
            args.obs_file = file
            name=f'{file}'
            args.results_dir = f"results/Inv_results_Ehlerobs/{name}"
            # Run the command
            run() 
            print(
                session_num,
                name,
                opti_regu_grad_ela,
                learning_rate,
            )
            session_num += 1
