import netCDF4
import numpy as np
from scipy import stats

# Load the NetCDF file
dataset = netCDF4.Dataset('alps_topo/ex.nc')

# Loop over each variable in the dataset
for var_name in dataset.variables:
    # Get the data for the variable
    var_data = dataset.variables[var_name][:]
    
    # Calculate the min, max, and most occurred value
    min_val = np.min(var_data)
    max_val = np.max(var_data)
    most_occurred = stats.mode(var_data.flatten())[0][0]
    
    # Print the results
    print(f'Variable: {var_name}')
    print(f'Min Value: {min_val}')
    print(f'Max Value: {max_val}')
    print(f'Most Occurred Value: {most_occurred}\n')
