import numpy as np
import os
from sklearn.model_selection import train_test_split
from   netCDF4 import Dataset
import tensorflow as tf


class DataLoader:
    def __init__(self,config):
        self.args =config
        
    def set_scaling(self):
        #Define the scaling of each variable
        
        self.naturalbounds = {}
        #
        self.naturalbounds["gfp"]         = 2730
        self.naturalbounds["topg"]   	  = 4550
        self.naturalbounds["ela"]   	  = 2320
        self.naturalbounds["time"]   	  = 5
        self.naturalbounds["A"]   	      = 78
        self.naturalbounds["beta"]   	  = 0.008
        self.naturalbounds["c"]   	      = 2.1
        
        return self.naturalbounds
    def findsubdata(self):
        """
            find the directory of data
        """
        subdatasetpath = [f.path for f in os.scandir(self.config.folder) if f.is_dir()]
        subdatasetpath.sort(key=lambda x: (os.path.isdir(x), x))  # sort alphabtically
        subdatasetname = [f.split("/")[-1] for f in subdatasetpath]
        return subdatasetname, subdatasetpath

    def load_netcdf(self,field_in,file_path=None):
        if file_path== None:
            file_path=os.path.join(self.args.data_dir,self.args.dataset)
            
        #initialize the field bounderies
        self.set_scaling()    
        
        # Load the netCDF file
        data = Dataset(os.path.join(file_path,self.args.obs_file))

        # Get the variable 'obs_gfp'
        obs_gfp = data.variables['max_thk_obs'][:]
        obs_gfp=obs_gfp.T
        time = data.variables['time'][:]
        time=time.T
        ela = data.variables['ela'][:]
        ela=ela.T
        topg = data.variables['topg'][:]
        topg=topg.T
        A = data.variables['A'][:]
        A=A.T
        beta = data.variables['beta'][:]
        beta=beta.T
        c = data.variables['c'][:]
        c=c.T

        return obs_gfp, time, ela, topg, A, beta, c
    
    def split_data_1(self, field_in, field_out, file_path=None, ratio=None):
    
        """first method to split data.
        Split the data in conventional way (80/20)

        Args:
            file_path (string): directory whre to store the data.
            field_in (list): list of features.
            field_out (list): list of torgets.
            ratio (float, optional): Ratia train/test. Defaults to 0.8.

        Returns:
            Train_(x,y) and test_(x,y) data ready to fit the model.
        """ 
        if file_path== None:
            file_path=os.path.join(self.args.data_dir,self.args.dataset)
        
        if ratio== None:
            ratio=self.args.train_test_ratio
        
        #initialize the field bounderies
        self.set_scaling()
        
        # Load the netCDF file
        data = Dataset(os.path.join(file_path, "ex.nc"))

        seed = 1234
        np.random.seed(seed)
        # Shuffle the data along the record dimension (idx)
        idx = data.dimensions['idx'].size
        shuffled_idx = np.random.permutation(idx)
        idx_= np.arange(idx)
        # Split the data into train and test sets with the specified ratio
        train_idx, test_idx = train_test_split(idx_, test_size=1-ratio,shuffle=False)

        # Split the train set into train_x and train_y
        train_x = np.transpose(np.array([data.variables[var][train_idx]/self.naturalbounds[var] for var in field_in]).T, (2, 0, 1, 3))
        train_y = np.transpose(np.array([data.variables[var][train_idx]/self.naturalbounds[var] for var in field_out]).T, (2, 0, 1, 3))

        # Split the test set into test_x and test_y
        test_x = np.transpose(np.array([data.variables[var][test_idx]/self.naturalbounds[var] for var in field_in]).T, (2, 0, 1, 3))
        test_y = np.transpose(np.array([data.variables[var][test_idx]/self.naturalbounds[var] for var in field_out]).T, (2, 0, 1, 3))

        return train_x, train_y, test_x, test_y


    def split_data_2(self,field_in, field_out, file_path=None, topo=None):
    
        """second method to split data.
        Splits data into test (one specific topo) and train ( all the other topos)

        Args:
            file_path (string): directory whre to store the data.
            topo (string): topography used to validate/test the model.
            field_in (list): list of features.
            field_out (list): list of torgets.

        Returns:
            Train_(x,y) and test_(x,y) data ready to fit the model.
        """ 
        if file_path== None:
            file_path=os.path.join(self.args.data_dir,self.args.dataset)
        
        if topo== None:
            topo=self.args.test_topo
            
        #initialize the field bounderies
        self.set_scaling()
        
        # Load the netCDF file
        data = Dataset(os.path.join(file_path, "ex.nc"))

        # Split the data into train and test sets with the specified topography
        test_idx = np.where(data.variables['name'][:] == topo)[0]
        train_idx = np.where(data.variables['name'][:] != topo)[0]

        # Split the train set into train_x and train_y
        train_x = np.transpose(np.array([data.variables[var][train_idx]/self.naturalbounds[var] for var in field_in]).T, (2, 0, 1, 3))
        train_y = np.transpose(np.array([data.variables[var][train_idx]/self.naturalbounds[var] for var in field_out]).T, (2, 0, 1, 3))

        # Split the test set into test_x and test_y
        test_x = np.transpose(np.array([data.variables[var][test_idx]/self.naturalbounds[var] for var in field_in]).T, (2, 0, 1, 3))
        test_y = np.transpose(np.array([data.variables[var][test_idx]/self.naturalbounds[var] for var in field_out]).T, (2, 0, 1, 3))

        return train_x, train_y, test_x, test_y
    
    def split_data_3(self,field_in, field_out, topo=None, file_path=None, ratio=None):
    
        """Third method to split data.
        Select a topo of your choice and split the data in conventional way (80/20) 

        Args:
            file_path (string): directory whre to store the data.
            field_in (list): list of features.
            field_out (list): list of torgets.
            ratio (float, optional): Ratia train/test. Defaults to 0.8.

        Returns:
            Train_(x,y) and test_(x,y) data ready to fit the model.
        """ 
        if file_path== None:
            file_path=os.path.join(self.args.data_dir,self.args.dataset)
        
        if ratio== None:
            ratio=self.args.train_test_ratio
        
        if topo== None:
            topo=self.args.test_topo
        
        #initialize the field bounderies
        self.set_scaling()
            
        # Load the netCDF file
        data = Dataset(os.path.join(file_path, "ex.nc"))
        # Split the data into train and test sets with the specified topography
        new_ds_idx = np.where(data.variables['name'][:] == topo)[0]
        print(data.variables['name'][9])
        
        seed = 1234
        np.random.seed(seed)
        # Shuffle the data along the record dimension (idx)
        shuffled_idx = np.random.permutation(new_ds_idx)
        
        # Split the data into train and test sets with the specified ratio
        train_idx, test_idx = train_test_split(shuffled_idx, test_size=1-ratio, random_state=seed)

        # Split the train set into train_x and train_y
        train_x = np.transpose(np.array([data.variables[var][train_idx]/self.naturalbounds[var] for var in field_in]).T, (2, 0, 1, 3))
        train_y = np.transpose(np.array([data.variables[var][train_idx]/self.naturalbounds[var] for var in field_out]).T, (2, 0, 1, 3))

        # Split the test set into test_x and test_y
        test_x = np.transpose(np.array([data.variables[var][test_idx]/self.naturalbounds[var] for var in field_in]).T, (2, 0, 1, 3))
        test_y = np.transpose(np.array([data.variables[var][test_idx]/self.naturalbounds[var] for var in field_out]).T, (2, 0, 1, 3))

        return train_x, train_y, test_x, test_y

    
    def split_data_4(self,field_in, field_out, file_path=None, time=None):
    
        """Forth method to split data.
        Splits data into test (one specific time) and train ( all the other times)

        Args:
            file_path (string): directory whre to store the data.
            topo (string): topography used to validate/test the model.
            field_in (list): list of features.
            field_out (list): list of torgets.

        Returns:
            Train_(x,y) and test_(x,y) data ready to fit the model.
        """ 
        if file_path== None:
            file_path=os.path.join(self.args.data_dir,self.args.dataset)
        
        if time== None:
            time=5
            
        #initialize the field bounderies
        self.set_scaling()
        
        # Load the netCDF file
        data = Dataset(os.path.join(file_path, "ex.nc"))

        # Split the data into train and test sets with the specified topography
        test_idx = np.where(data.variables['time'][:,1,1] == time)[0]
        train_idx = np.where(data.variables['time'][:,1,1] != time)[0]
         
            
        # Split the train set into train_x and train_y
        train_x = np.transpose(np.array([data.variables[var][train_idx]/self.naturalbounds[var] for var in field_in]).T, (2, 0, 1, 3))
        train_y = np.transpose(np.array([data.variables[var][train_idx]/self.naturalbounds[var] for var in field_out]).T, (2, 0, 1, 3))

        # Split the test set into test_x and test_y
        test_x = np.transpose(np.array([data.variables[var][test_idx]/self.naturalbounds[var] for var in field_in]).T, (2, 0, 1, 3))
        test_y = np.transpose(np.array([data.variables[var][test_idx]/self.naturalbounds[var] for var in field_out]).T, (2, 0, 1, 3))
        
        

        return train_x, train_y, test_x, test_y
    def split_data_5(self,field_in, field_out, file_path=None, time=None):
    
        """fifth method to split data.
        Splits data into test and train after it exludes specific time.

        Args:
            file_path (string): directory whre to store the data.
            topo (string): topography used to validate/test the model.
            field_in (list): list of features.
            field_out (list): list of torgets.

        Returns:
            Train_(x,y) and test_(x,y) data ready to fit the model.
        """ 
        if file_path== None:
            file_path=os.path.join(self.args.data_dir,self.args.dataset)
        
        if time== None:
            time=6
        ratio =0.8
          
        #initialize the field bounderies
        self.set_scaling()
        
        # Load the netCDF file
        data = Dataset(os.path.join(file_path, "ex.nc"))

        # Split the data into train and test sets with the specified topography
        new_ds_idx = np.where(data.variables['time'][:,1,1] != time)[0]
        seed = 1234
        np.random.seed(seed)
        # Shuffle the data along the record dimension (idx)
        shuffled_idx = np.random.permutation(new_ds_idx) 
            
        # Split the data into train and test sets with the specified ratio
        train_idx, test_idx = train_test_split(shuffled_idx, test_size=1-ratio, random_state=seed)

        # Split the train set into train_x and train_y
        train_x = np.transpose(np.array([data.variables[var][train_idx]/self.naturalbounds[var] for var in field_in]).T, (2, 0, 1, 3))
        train_y = np.transpose(np.array([data.variables[var][train_idx]/self.naturalbounds[var] for var in field_out]).T, (2, 0, 1, 3))

        # Split the test set into test_x and test_y
        test_x = np.transpose(np.array([data.variables[var][test_idx]/self.naturalbounds[var] for var in field_in]).T, (2, 0, 1, 3))
        test_y = np.transpose(np.array([data.variables[var][test_idx]/self.naturalbounds[var] for var in field_out]).T, (2, 0, 1, 3))

        return train_x, train_y, test_x, test_y
#----------------------------------------------------------------
