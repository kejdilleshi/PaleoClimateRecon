import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import tensorflow as tf
import numpy as np

class ModelEvaluator:
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
        
        # Plot the true data and the model's predictions side by side for each test sample
        for i in range(len(test_x[:nb_test])):
            self.plotresu_f(os.path.join(self.args.results_dir,'Evaluation'), predictions[i]*self.naturalbounds["gfp"], test_y[i]*self.naturalbounds["gfp"], i)
        
    def plot_lc(self,history):
        #scale back the loss values
        
        scale_factor=self.naturalbounds["gfp"]
        train_loss = history.history['loss'] * scale_factor
        val_loss = history.history['val_loss'] * scale_factor
        
        # Plot training & validation loss values
        plt.figure(figsize=(20, 10))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.savefig(os.path.join(self.args.results_dir,"Lr_.png"))

    def plotresu_f(self, pathofresults, pred_outputs, true_outputs, idx):
        
        m1 = tf.keras.metrics.MeanAbsoluteError()
        mae = m1(pred_outputs, true_outputs)

        m2 = tf.keras.metrics.RootMeanSquaredError()
        mse = m2(pred_outputs, true_outputs)

        pred_outputs0 = np.linalg.norm(pred_outputs, axis=2)
        true_outputs0 = np.linalg.norm(true_outputs, axis=2)

        valmaxx = max(
            [np.quantile(pred_outputs0, 0.999), np.quantile(true_outputs0, 0.999)]
        )

        fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(20, 10))

        ax1.set_title("PREDICTED, mae: %.5f, mse: %.5f" % (mae, mse))
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

        ax2.set_title("TRUE")
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

        plt.tight_layout()
        
         # Save the figure for this test sample
        plt.savefig(os.path.join(pathofresults,"evaluation_{}.png".format(idx)))
        
    