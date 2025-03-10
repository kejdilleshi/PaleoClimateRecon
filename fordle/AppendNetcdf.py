from netCDF4 import Dataset
import numpy as np

# Open the existing netCDF file
path=f'alps_transfer_A/ex.nc'
nc = Dataset(path, 'a')  # 'a' stands for append mode

# Get the dimensions
x_dim = len(nc.dimensions['x'])
y_dim = len(nc.dimensions['y'])
idx_dim = len(nc.dimensions['idx'])

# Create new variables
nc.createVariable('A', 'f8', ('idx', 'x', 'y'))
nc.createVariable('beta', 'f8', ('idx', 'x', 'y'))
nc.createVariable('c', 'f8', ('idx', 'x', 'y'))

c_values=[39,156]

# Assign values to the new variables
# nc.variables['A'][:] = np.full((idx_dim, x_dim, y_dim), 78)
nc.variables['beta'][:] = np.full((idx_dim, x_dim, y_dim), 0.008)
nc.variables['c'][:] = np.full((idx_dim, x_dim, y_dim), 2.1)
# Assign values to the new variable 'c'
for i in range(idx_dim):
    c_value = c_values[i // 50 % len(c_values)]
    nc.variables['A'][i,:,:] = np.full((x_dim, y_dim),c_value)  # Replace all_data[i][3] with your actual value

# Close the file

nc.close()

print("----------------Data updated!----------------")
