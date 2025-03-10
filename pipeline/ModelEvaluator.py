import matplotlib.pyplot as plt
import os
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import tensorflow as tf
import numpy as np

class ModelEvaluator:
    cm=1/2.54
    def __init__(self,config,naturalbounds):
        self.args = config
        self.naturalbounds=naturalbounds
    #Save model parameters
        
    def save_hparams(self,run_name, hparams):
        with open(os.path.join(self.args.results_dir,'hparams.txt'), 'a') as f:
            f.write(f'--- Session number: {run_name}\n')
            f.write(f'Result directory: {self.args.results_dir}\n')
            f.write(f'{hparams}\n')
            
           
    def evaluate(self, model, test_x, test_y,nb_test=-1):
        
        # Get the model's predictions on the test data
        predictions = model.predict(test_x)
        
        # Create results directory
        os.makedirs(self.args.results_dir, exist_ok=True)
        # Create directory to save figures. 
        os.makedirs(os.path.join(self.args.results_dir,'Evaluation'), exist_ok=True)
        
        # If nb_test is -1, use all test samples, otherwise use nb_test samples
        if nb_test == -1:
            nb_test = len(test_x)

        # Generate a list of random indices
        indices = np.random.choice(len(test_x), size=nb_test, replace=False)
        columns = [f"Region_{i+1}" for i in range(9)]  # Naming regions
        df = pd.DataFrame( columns=columns)
        # Plot the true data and the model's predictions side by side for each test sample
        for idx in indices:
            
            difference=self.plotresu_g(os.path.join(self.args.results_dir,'Evaluation'), test_x, predictions[idx]*self.naturalbounds["gfp"], test_y[idx]*self.naturalbounds["gfp"], idx)
            self.add_gfp_to_dataframe(difference,df)
            # self.plotresu_f(os.path.join(self.args.results_dir,'Evaluation'), predictions[idx]*self.naturalbounds["gfp"], test_y[idx]*self.naturalbounds["gfp"], idx)
        df.to_csv(os.path.join(self.args.results_dir,'mean_values.csv'), index=False)
	        
    def get_mean(self,GFP):
        regions = [
            (50, 75, 90, 100),
            (110, 120, 120, 150),
            (130, 150, 220, 260),
            (160, 170, 310, 350),
            (220, 230, 310, 350),
            (230, 240, 230, 270),
            (220, 230, 180, 210),
            (180, 195, 100, 120),
            (120, 140, 50, 80),
        ]
        return [np.mean(GFP[x1:x2, y1:y2]) for x1, x2, y1, y2 in regions]
    # Function to add a new GFP array to the DataFrame
    def add_gfp_to_dataframe(self,GFP, df):
        mean_values = self.get_mean(GFP)
        df.loc[len(df)] = mean_values  # Add a new row to the DataFrame
        return df
    def plot_lc(self, history):
        # Verify scale_factor
        # print("Scale factor:", self.naturalbounds["gfp"])  # Make sure this prints a numeric value
        
        # Check the first few loss values
        # print("First 5 original training losses:", history.history['loss'][:5])
        # print("First 5 original validation losses:", history.history['val_loss'][:5])
        
        # Convert cm to inches for figsize
        width_in_inches = 8.5 * self.cm
        height_in_inches = 4 * self.cm
        
        scale_factor = self.naturalbounds["gfp"]
        
        # Ensure scale_factor is numeric and not zero
        if isinstance(scale_factor, (int, float)) and scale_factor != 0:
            # Scale the loss values
            train_loss = [x * scale_factor for x in history.history['loss']]
            val_loss = [x * scale_factor for x in history.history['val_loss']]
        else:
            print("Error with scale factor:", scale_factor)
            # Default to unscaled values if there is an issue
            train_loss = history.history['loss']
            val_loss = history.history['val_loss']
        
        # Plot training & validation loss values
        fig, axs = plt.subplots(1, 1, figsize=(width_in_inches, height_in_inches))
        axs.plot(np.arange(len(train_loss)), train_loss, label='Training')
        axs.plot(np.arange(len(val_loss)), val_loss, label='Validation')
        axs.set_xlabel('Epoch',fontsize=8)
        axs.set_ylabel('Loss [m]',fontsize=8)
        axs.legend(fontsize=6)
        
        # Debugging print
        # print("Scaled loss (first 5):", train_loss[:5])
        
        plt.savefig(os.path.join(self.args.results_dir, 'Lr_'), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    def plot_lc_normalized(self, history):
      
        # Convert cm to inches for figsize
        width_in_inches = 8.5 * self.cm
        height_in_inches = 4 * self.cm
        
        # Default to unscaled values if there is an issue
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        # Plot training & validation loss values
        fig, axs = plt.subplots(1, 1, figsize=(width_in_inches, height_in_inches))
        axs.plot(np.arange(len(train_loss)), train_loss, label='Training')
        axs.plot(np.arange(len(val_loss)), val_loss, label='Validation')
        axs.set_xlabel('Epoch',fontsize=8)
        axs.set_ylabel('Loss',fontsize=8)
        axs.legend(fontsize=6)
        
        plt.savefig(os.path.join(self.args.results_dir, 'Lr_normalized'), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
    def plotresu_f(self, pathofresults, pred_outputs, true_outputs, idx):
        m1 = tf.keras.metrics.MeanAbsoluteError()
        mae = m1(pred_outputs, true_outputs)
    
        m2 = tf.keras.metrics.RootMeanSquaredError()
        mse = m2(pred_outputs, true_outputs)
    
        pred_outputs0 = np.linalg.norm(pred_outputs, axis=2)
        true_outputs0 = np.linalg.norm(true_outputs, axis=2)
    
        # Rotate the predicted and true outputs by 90 degrees counterclockwise
        pred_outputs0 = np.rot90(pred_outputs0,k=3)
        true_outputs0 = np.rot90(true_outputs0,k=3)
    
        # Flip the predicted and true outputs left to right
        pred_outputs0 = np.fliplr(pred_outputs0)
        true_outputs0 = np.fliplr(true_outputs0)
    
        # Calculate the difference between the true and predicted outputs
        diff_outputs0 = (true_outputs0 - pred_outputs0)
    
        valmaxx = max(
            [np.quantile(pred_outputs0, 0.999), np.quantile(true_outputs0, 0.999)]
        )
    
        # Find the maximum absolute value in the difference
        max_diff = np.max(np.abs(diff_outputs0))
    
        fig, (ax2, ax1, ax3) = plt.subplots(1,3, figsize=(17.5*self.cm,5*self.cm))
    
        ax1.set_title("PREDICTED",fontdict={'fontsize':14, 'fontweight': 'bold'})
        im1 = ax1.imshow(
            pred_outputs0,
            origin="lower",
            vmin=0,
            vmax=valmaxx,
            cmap=cm.get_cmap("viridis", 10),
        )
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar1 = plt.colorbar(im1, format="%.2f", cax=cax1)
        ax1.axis("off")
    
        ax2.set_title("TRUE",fontdict={'fontsize':14, 'fontweight': 'bold'})
        im2 = ax2.imshow(
            true_outputs0,
            origin="lower",
            vmin=0,
            vmax=valmaxx,
            cmap=cm.get_cmap("viridis", 10),
        )
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar2 = plt.colorbar(im2, format="%.2f", cax=cax2)
        ax2.axis("off")
    
        # Plot the difference between the true and predicted outputs
        ax3.set_title("DIFFERENCE, mae: %.5f, mse: %.5f" % (mae, mse),fontdict={'fontsize':14, 'fontweight': 'bold'})
        im3 = ax3.imshow(
            diff_outputs0,
            origin="lower",
            vmin=-max_diff,  # Set vmin to negative of max_diff
            vmax=max_diff,  # Set vmax to max_diff
            cmap=cm.get_cmap("seismic"),
        )
        divider = make_axes_locatable(ax3)
        cax3 = divider.append_axes("right", size="5%", pad=0.05)
        cbar3 = plt.colorbar(im3, format="%.2f", cax=cax3)
        ax3.axis("off")
    
        plt.tight_layout()
    
         # Save the figure for this test sample
        plt.savefig(os.path.join(pathofresults,"evaluation_{}.png".format(idx)))
        plt.close(fig)
    
    def plotresu_g(self, pathofresults, test_x, pred_outputs, true_outputs, idx):
        # Create meshgrid for plotting with pcolor
        x = np.linspace(0, 1, 451)
        y = np.linspace(0, 1, 301)
        X, Y = np.meshgrid(x, y)

        # Calculate the difference between the true and predicted outputs
        diff_outputs = true_outputs - pred_outputs
        # Calculate the relative difference and its mean
        relative_diff = np.mean(np.abs((true_outputs - pred_outputs) / true_outputs))

        # Calculate the mean time from test_x
        mean_time = np.mean(test_x[idx, :, :, 0] * self.naturalbounds['time'])

        # Rotate and flip the fields
        ela_field = np.fliplr(np.rot90(test_x[idx, :, :, 1] * self.naturalbounds['ela'], k=3))
        pred_outputs = np.fliplr(np.rot90(pred_outputs, k=3))
        true_outputs = np.fliplr(np.rot90(true_outputs, k=3))
        diff_outputs = np.fliplr(np.rot90(diff_outputs, k=3))

        # Find the maximum absolute value in the difference
        max_diff = np.max(np.abs(diff_outputs)) * 0.8

        # Create the figure and subplots
        fig, axes = plt.subplots(2, 2, figsize=(8.7 * self.cm, 6.25 * self.cm))
        ax1, ax2, ax3, ax4 = axes.flatten()

        # Plot ELA field (Subfigure A)
        ax1.set_title("ELA", fontdict={'fontsize': 8})
        im1 = ax1.pcolor(X, Y, np.squeeze(ela_field / 100))
        cbar1 = plt.colorbar(im1, ax=ax1, format="%d", fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=6)
        cbar1.ax.set_ylabel('Altitude/100 [m]', fontsize=6, rotation=270, labelpad=5)
        ax1.set_aspect(301/451, adjustable='box')
        ax1.axis("off")
        ax1.text(0.05, 0.95, "A", transform=ax1.transAxes, fontsize=6, va='top', ha='left')

        # Plot predicted outputs (Subfigure B)
        ax2.set_title("Predicted", fontdict={'fontsize': 8})
        im2 = ax2.pcolor(X, Y, np.squeeze(pred_outputs / 100), cmap='Blues', shading='auto')
        ax2.contourf(X, Y, np.squeeze(true_outputs), [5, 10], cmap='gray')
        cbar2 = plt.colorbar(im2, ax=ax2, format="%d", fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=6)
        cbar2.ax.set_ylabel('Thickness/100 [m]', fontsize=6, rotation=270, labelpad=5)
        ax2.set_aspect(301/451, adjustable='box')
        ax2.axis("off")
        ax2.text(0.05, 0.95, "B", transform=ax2.transAxes, fontsize=6, va='top', ha='left')

        # Plot the difference between true and predicted outputs (Subfigure C)
        ax3.set_title(f"True-Predicted", fontdict={'fontsize': 8})
        im3 = ax3.pcolor(X, Y, np.squeeze(diff_outputs), vmin=-max_diff, vmax=max_diff, cmap=cm.get_cmap("seismic"))
        ax3.contourf(X, Y, np.squeeze(true_outputs), [5, 10], cmap='gray')
        cbar3 = plt.colorbar(im3, ax=ax3, format="%d", fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=6)
        cbar3.ax.set_ylabel('Thickness [m]', fontsize=6, rotation=270, labelpad=5)
        ax3.set_aspect(301/451, adjustable='box')
        ax3.axis("off")
        ax3.text(0.05, 0.95, "C", transform=ax3.transAxes, fontsize=6, va='top', ha='left')

        # Plot true outputs (Subfigure D)
        ax4.set_title("True", fontdict={'fontsize': 8})
        im4 = ax4.pcolor(X, Y, np.squeeze(true_outputs / 100), cmap='Blues', shading='auto')
        ax4.contourf(X, Y, np.squeeze(true_outputs), [5, 10], cmap='gray')
        cbar4 = plt.colorbar(im4, ax=ax4, format="%d", fraction=0.046, pad=0.04)
        cbar4.ax.tick_params(labelsize=6)
        cbar4.ax.set_ylabel('Thickness/100 [m]', fontsize=6, rotation=270, labelpad=5)
        ax4.set_aspect(301/451, adjustable='box')
        ax4.axis("off")
        ax4.text(0.05, 0.95, "D", transform=ax4.transAxes, fontsize=6, va='top', ha='left')

        # Adjust layout and add supertitle
        fig.subplots_adjust(wspace=0.3, hspace=0.2)
        plt.suptitle(f"Time of Simulation: {int(mean_time)} k years", fontsize=8, y=1.02)

        # Save the figure for this test sample
        plt.savefig(os.path.join(pathofresults, f"evaluation_{idx}.png"), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()
        plt.close(fig)

        return diff_outputs