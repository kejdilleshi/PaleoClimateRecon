import os
import netCDF4 as nc

# Define the folder containing the NetCDF files and the new data file
folder_path = "alps_topo"
new_data_file =  "obs_Ehler_2D.nc"#"obs_Jouvet2024.nc"

# Read the new data from the 'obs_Jouvet2024.nc' file
with nc.Dataset(new_data_file) as new_nc:
    new_max_obs_thk = new_nc.variables["max_thk_obs"][:]

# Iterate through all NetCDF files in the folder starting with 'obs_'
for file_name in os.listdir(folder_path):
    if file_name.startswith("obs_") and file_name.endswith(".nc"):
        file_path = os.path.join(folder_path, file_name)

        # Open the current NetCDF file in write mode
        with nc.Dataset(file_path, "a") as nc_file:
            # Check if 'max_obs_thk' variable exists in the file
            if "max_thk_obs" in nc_file.variables:
                # Replace the values of 'max_obs_thk' with the new data
                nc_file.variables["max_thk_obs"][:] = new_max_obs_thk
                print(f"Replaced 'max_obs_thk' values in {file_name}")
            else:
                print(f"'max_obs_thk' variable not found in {file_name}")
