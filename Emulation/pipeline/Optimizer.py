import sys
sys.path.append('/work/FAC/FGSE/IDYST/gjouvet/default/klleshi/EmulatorDS')
from pipeline import DataLoader
import tensorflow as tf
from read_config_param import get_args
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import os
 
           
class Optimizer:
    def __init__(self, args):
        self.args = args
        self.cost_ela_values = []
        self.regu_ela_values = []
        self.regu_time_values = []
        self.tot_cost = []
        self.field_in = ['time', 'ela', 'topg']
        self.field_out = ['gfp']
        self.data_loader = DataLoader(args) 
        self.naturalbounds = self.data_loader.set_scaling()
        self.obs, self.time, self.ela, self.topg = self.data_loader.load_netcdf(self.field_in)
        self.iceflow_model = tf.keras.models.load_model(self.args.modeldir)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.learning_rate)
        self.topg=tf.Variable(self.topg /self.naturalbounds["topg"],dtype=tf.float32) 
        self.ela = tf.Variable(self.ela / self.naturalbounds["ela"],dtype=tf.float32)  
        self.time = tf.Variable(self.time / self.naturalbounds["time"],dtype=tf.float32) 


    def plot_ela_time_max_thk_calc(self,ela, time, max_thk_calc, i):

        # Create results directory
        os.makedirs(self.args.results_dir, exist_ok=True)
        # Rotate the predicted and true outputs by 90 degrees counterclockwise
        ela = np.rot90(ela.numpy(),k=3)
        time = np.rot90(time.numpy(),k=3)
        max_thk_calc = np.rot90(max_thk_calc.numpy(),k=3)

        # Flip the predicted and true outputs left to right
        ela = np.fliplr(ela)
        time = np.fliplr(time)
        max_thk_calc = np.fliplr(max_thk_calc)

        fig, (ax2, ax1, ax3) = plt.subplots(1,3, figsize=(30,10))


        ax1.set_title(f'Iteration {i}: ELA')
        im1 = ax1.imshow(
            ela*self.naturalbounds['ela'],
            origin="lower",
            cmap=cm.get_cmap("viridis", 15),
        )
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar1 = plt.colorbar(im1, format="%.2f", cax=cax1)
        ax1.axis("off")


        ax2.set_title(f'Iteration {i}: Time')
        im2 = ax2.imshow(
            time*self.naturalbounds['time'],
            origin="lower",
            cmap=cm.get_cmap("viridis", 15),
        )
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar2 = plt.colorbar(im2, format="%.2f", cax=cax2)
        ax2.axis("off")

        ax3.set_title(f'Iteration {i}: Max Thk Calc')
        im3 = ax3.imshow(
            max_thk_calc,
            origin="lower",
            cmap=cm.get_cmap("viridis", 10),
        )
        divider = make_axes_locatable(ax3)
        cax3 = divider.append_axes("right", size="5%", pad=0.05)
        cbar3 = plt.colorbar(im3, format="%.2f", cax=cax3)
        ax3.axis("off")
        plt.tight_layout()

         # Save the figure for this test sample
        plt.savefig(f'{self.args.results_dir}/iteration_{i}.png')
        plt.close(fig)




    def apply_second_order_regularization(self,ela, time, args):
        # Compute first order spatial gradients
        dedx1 = (ela[:,1:] - ela[:,:-1]) / 1
        dedy1 = (ela[1:,:] - ela[:-1,:]) / 1

        # Compute second order spatial gradients
        dedx2 = (dedx1[:,1:] - dedx1[:,:-1]) / 1
        dedy2 = (dedy1[1:,:] - dedy1[:-1,:]) / 1

        # Compute regularization term for ela
        REGU_ela = args.opti_regu_grad_ela * (tf.nn.l2_loss(dedx2) + tf.nn.l2_loss(dedy2))

        # Compute first order spatial gradients for time
        dedx1 = (time[:,1:] - time[:,:-1]) / 1
        dedy1 = (time[1:,:] - time[:-1,:]) / 1

        # Compute second order spatial gradients for time
        dedx2 = (dedx1[:,1:] - dedx1[:,:-1]) / 1
        dedy2 = (dedy1[1:,:] - dedy1[:-1,:]) / 1

        # Compute regularization term for time
        REGU_time = args.opti_regu_grad_time * (tf.nn.l2_loss(dedx2) + tf.nn.l2_loss(dedy2))

        return REGU_ela, REGU_time


    def apply_fourth_order_regularization(self, ela, time, args):
        # Compute first order spatial gradients
        dedx1 = (ela[:,1:] - ela[:,:-1]) / 1
        dedy1 = (ela[1:,:] - ela[:-1,:]) / 1

        # Compute second order spatial gradients
        dedx2 = (dedx1[:,1:] - dedx1[:,:-1]) / 1
        dedy2 = (dedy1[1:,:] - dedy1[:-1,:]) / 1

        # Compute third order spatial gradients
        dedx3 = (dedx2[:,1:] - dedx2[:,:-1]) / 1
        dedy3 = (dedy2[1:,:] - dedy2[:-1,:]) / 1

        # Compute fourth order spatial gradients
        dedx4 = (dedx3[:,1:] - dedx3[:,:-1]) / 1
        dedy4 = (dedy3[1:,:] - dedy3[:-1,:]) / 1

        # Compute regularization term for ela
        REGU_ela = args.opti_regu_grad_ela * (tf.nn.l2_loss(dedx4) + tf.nn.l2_loss(dedy4))

        # Compute first order spatial gradients for time
        dedx1 = (time[:,1:] - time[:,:-1]) / 1
        dedy1 = (time[1:,:] - time[:-1,:]) / 1

        # Compute second order spatial gradients for time
        dedx2 = (dedx1[:,1:] - dedx1[:,:-1]) / 1
        dedy2 = (dedy1[1:,:] - dedy1[:-1,:]) / 1

        # Compute third order spatial gradients for time
        dedx3 = (dedx2[:,1:] - dedx2[:,:-1]) / 1
        dedy3 = (dedy2[1:,:] - dedy2[:-1,:]) / 1

        # Compute fourth order spatial gradients for time
        dedx4 = (dedx3[:,1:] - dedx3[:,:-1]) / 1
        dedy4 = (dedy3[1:,:] - dedy3[:-1,:]) / 1

        # Compute regularization term for time
        REGU_time = args.opti_regu_grad_time * (tf.nn.l2_loss(dedx4) + tf.nn.l2_loss(dedy4))

        return REGU_ela, REGU_time
    
    def save_hparams(self,l1,l2):
        # Create results directory
        os.makedirs(self.args.results_dir, exist_ok=True)
        
        with open(os.path.join(self.args.results_dir,'hparams.txt'), 'a') as f:
            f.write(f'--- Session number: \n')
            f.write(f'Result directory: {self.args.results_dir}\n')
            f.write(f'opti_regu_grad_time: {self.args.opti_regu_grad_time}, opti_regu_grad_ela: {self.args.opti_regu_grad_ela}, learning_rate : {self.args.learning_rate}\n')
            f.write(f'L1: {l1}, L2: {l2}\n')
            
    def plot_curves(self,cost_ela_values, regu_ela_values, regu_time_values,tot_cost):
        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the curves
        ax.plot(cost_ela_values, label='COST_ela')
        ax.plot(regu_ela_values, label='REGU_ela')
        ax.plot(regu_time_values, label='REGU_time')
        ax.plot(tot_cost, label='Cost')




        # Add labels and a legend
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.set_title('Curve Plot')
        ax.legend()

        # Save the plot to a file (e.g., in PNG format)
        save_path = os.path.join(self.args.results_dir, 'curves_plot.png')
        fig.savefig(save_path)

            
    def optimize(self):
        # Perturbate the 'observations' as all measurements have uncertainties. We include 10% uncertainties. 
        # ACT = tf.math.not_equal(self.obs, 0)
        
        # self.obs[ACT]=self.obs[ACT]+np.random.normal(self.obs[ACT],np.mean(self.obs[ACT])*0.1)

        for i in range(self.args.opti_nbitmax+1):
            with tf.GradientTape() as t:
                t.watch(self.ela)
                t.watch(self.time)
                X= tf.concat( [tf.expand_dims(self.time,axis=-1),
                                 tf.expand_dims(self.ela,axis=-1),
                                 tf.expand_dims(self.topg,axis=-1),
                                ],axis=-1)
                X = tf.expand_dims(X, axis=0)
                Y=self.iceflow_model(X)

                max_thk_calc = Y[0, :, :, 0]*self.naturalbounds["gfp"]
                ACT = ~tf.math.is_nan(self.obs)
                         
                # Define regularization parameters
                self.args.l1_lambda =1.0  # Weight for L1 norm
                self.args.l2_lambda = 1.0  # Weight for L2 norm
                
                # Calculate L1 and L2 norms
                l1_norm = tf.reduce_sum(tf.abs(self.obs[ACT] - max_thk_calc[ACT]))
                l2_norm = tf.reduce_sum(tf.square(self.obs[ACT] - max_thk_calc[ACT]))
                
                COST_ela = self.args.l1_lambda* l1_norm + self.args.l2_lambda * l2_norm

                REGU_ela,REGU_time=self.apply_second_order_regularization(self.ela, self.time,self.args)
                
                COST_min = 10 ** 10 * tf.math.reduce_sum(
                        tf.where(self.ela*self.naturalbounds['ela'] > 1000, 0.0, (self.ela*self.naturalbounds['ela']-1000)**2)
                    )
                
                COST_min_time = 10 ** 10 * tf.math.reduce_sum(
                        tf.where(self.time*self.naturalbounds['time'] > 1, 0.0, (self.time*self.naturalbounds['time']-1)**2)
                    )


                COST = COST_ela + REGU_ela + REGU_time + COST_min #+ COST_min_time

                var_to_opti = []
                var_to_opti.append(self.ela)
                # var_to_opti.append(self.time)

                grads = tf.Variable(t.gradient(COST, var_to_opti))
                self.optimizer.apply_gradients(
                    zip([grads[i] for i in range(grads.shape[0])], var_to_opti)
                )
                
                # get back optimized variables in the pool of self.variables
                self.ela.assign(self.ela)
                # self.time.assign(self.time)

                # Append the current values of COST_ela, REGU_ela, and REGU_time to the lists
                self.cost_ela_values.append(COST_ela.numpy())
                self.regu_ela_values.append(REGU_ela.numpy())
                self.regu_time_values.append(REGU_time.numpy())
                self.tot_cost.append(COST.numpy())
                
                if i % self.args.save_freq == 0 and i>1000:
                    print("Itteration {} : COST :{},\t{},\t{},\t{},\t{}".format(i, COST,COST_ela,REGU_ela,REGU_time,COST_min))
                    self.plot_ela_time_max_thk_calc(self.ela, self.time, max_thk_calc, i)
        
        self.save_hparams(l1=l1_norm,l2=l2_norm)           
        np.save(os.path.join(self.args.results_dir,'TIME.npy'),(self.time*self.naturalbounds['time']).numpy())
        np.save(os.path.join(self.args.results_dir,'ELA.npy'),(self.ela*self.naturalbounds['ela']).numpy())
        np.save(os.path.join(self.args.results_dir,'GFP.npy'),(max_thk_calc).numpy())

        # plt.figure()
        # plt.hist(self.obs.ravel())
        # plt.savefig('obs.png')
        # plt.show()
        
        
        self.plot_curves(self.cost_ela_values,self.regu_ela_values,self.regu_time_values,self.tot_cost)


    def NewtonOptimizer(self):
        optimizer = tf.optimizers.SGD(learning_rate=self.args.learning_rate)

        for i in range(self.args.opti_nbitmax+1):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(self.ela)
                # tape.watch(self.time)
                X= tf.concat( [tf.expand_dims(self.time,axis=-1),
                                 tf.expand_dims(self.ela,axis=-1),
                                 tf.expand_dims(self.topg,axis=-1),
                                ],axis=-1)
                X = tf.expand_dims(X, axis=0)
                Y=self.iceflow_model(X)

                max_thk_calc = Y[0, :, :, 0]*self.naturalbounds["gfp"]
                ACT = ~tf.math.is_nan(self.obs)
                
                
                # Define regularization parameters
                self.args.l2_lambda = 1.0  # Weight for L2 norm
                
                # Calculate L1 and L2 norms
                l2_norm = tf.reduce_sum(tf.square(self.obs[ACT] - max_thk_calc[ACT]))
                
                
                COST_ela = self.args.l2_lambda * l2_norm

                REGU_ela,REGU_time=self.apply_second_order_regularization(self.ela, self.time,self.args)
                
                COST_min = 10 ** 10 * tf.math.reduce_sum(
                        tf.where(self.ela*self.naturalbounds['ela'] > 1000, 0.0, (self.ela*self.naturalbounds['ela']-1000)**2)
                    )
                
                COST_min_time = 10 ** 10 * tf.math.reduce_sum(
                        tf.where(self.time*self.naturalbounds['time'] > 1, 0.0, (self.time*self.naturalbounds['time']-1)**2)
                    )


                COST = COST_ela + REGU_ela + REGU_time + COST_min #+ COST_min_time

                var_to_opti = []
                var_to_opti.append(self.ela)
                # var_to_opti.append(self.time)

            grads = tf.Variable(tape.gradient(COST, var_to_opti))
            hess=tf.Variable(tape.jacobian(grads,var_to_opti))
            update = tf.linalg.solve(hess, grads)
            optimizer.apply_gradients([(update,var_to_opti)])
            
            
            # get back optimized variables in the pool of self.variables
            self.ela.assign(self.ela)
            # self.time.assign(self.time)
        # np.save(os.path.join(self.args.results_dir,'TIME.npy'),(self.time*self.naturalbounds['time']).numpy())
        np.save('grad.npy', grads)
        np.save('hess.npy', hess)

        
        
        del tape  # delete the tape after using it
