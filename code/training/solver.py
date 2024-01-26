"""
Solver.py

@description: Solver class that implements the logic to train a neural network and plot 
              its learnign curves.
"""


import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator
from keras.losses import BinaryCrossentropy
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold                 
        
from training.callbacks import CustomBestEarlyStopping

# Optimization direction based on target metric
MODE_MAPPING = {'loss': 'min', 'val_loss': 'min', 'mean_euclidean_error': 'min', 'accuracy': 'max', 'val_mean_euclidean_error': 'min', 'val_accuracy': 'max'}


class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training a classification/regression model.
    It performs gradient descent using the given learning rate.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification/regression metric on both training and validation
    data to watch out for overfitting.

    After training (i.e. when train() method returns), the best performing weights are loaded back into
    the model, and the best achieved stats are stored in best_model_stats. History records for train loss,
    val loss, train metric, and val metric are also memorized.
    """
    
    def __init__(self, model: any, x_train: np.ndarray, y_train: np.ndarray, x_val=None, y_val=None, target='accuracy', **kwargs):
        """
        Construct a new Solver instance.

        Args:
            - model: a model object
            - x_train: training data
            - y_train: training ground-truth labels
        
        Optional:
            - x_val: validation data
            - y_val: validation ground-truth labels
            - target: metric to guide early-stopping
        """
        
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.validation_data = None
        
        if target not in MODE_MAPPING:
            raise ValueError(
                "Unknown <target> parameter, can only be 'loss/val_loss', 'accuracy/val_accuracy', or 'mean_euclidean_error/val_mean_euclidean_error'!"
            )
        self.mode = MODE_MAPPING[target]
        self.target = target
        
        
        self.validation_data = None if x_val is None else (self.x_val, self.y_val)
        
        self._reset()
    
    def _reset(self):
        """
        Reset/Setup some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Setup/reset some variables for book-keeping
        self.best_model_stats = None
        self.best_params = None

        self.train_loss_history = []
        self.val_loss_history = []
        
        self.train_metric_history = []
        self.val_metric_history = []
        
    def plot_history(self, out_path: str, loss: str, metric: str):
        """Plot learning curves for the model.

        Args:
            - out_path: output path to save the plots
            - loss: string title of y-axis (1st plot)
            - metric: string title of y-axis (2nd plot)
        """
        
        epochs = list(range(3, len(self.train_loss_history) + 1))

        plt.figure(figsize=(12, 5))

        # Plotting MSE loss
        ax1 = plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot = Losses
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.plot(epochs, self.train_loss_history[2:], label='Training', linestyle=':')
        ax1.plot(epochs, self.val_loss_history[2:], label='Test', linestyle='-',  linewidth=1)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel(loss)
        ax1.legend()
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Plotting metric (Accuracy or MEE)
        ax2 = plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot = metric
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.plot(epochs, self.train_metric_history[2:], label='Training', linestyle=':')
        ax2.plot(epochs, self.val_metric_history[2:], label='Test', linestyle='-', linewidth=1)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(metric)
        ax2.legend()
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.show()

    def train(self, epochs=50, batch_size=32, patience=None, lr_scheduler=None):
        """
        Train a model.
        
        Optional:
            epochs: number of epochs for training
            batch_size: size of the batch
            patience: number of epochs to wait for improvement before early-stopping.
            lr_scheduler: a custom callback for warm-up learning rate.
        """
    
        callbacks = []
    
        # Set learning rate scheduler
        if lr_scheduler is not None:
            callbacks.append(lr_scheduler)
        
        # Set Early-stopping
        if patience:
            early_stopping = CustomBestEarlyStopping(monitor=self.target, patience=patience, mode=self.mode, verbose=1, restore_best_weights=True)
            callbacks.append(early_stopping)

        # Train
        history = self.model.fit(self.x_train, self.y_train,
                                 validation_data=self.validation_data,
                                 epochs=epochs, batch_size=batch_size, callbacks=callbacks, shuffle=True, verbose=1)
        
        # Record loss history
        self.train_loss_history = history.history['loss']
        self.val_loss_history = history.history.get('val_loss', [])
        
        # Record metric history
        self.train_metric_history = history.history[self.model.metrics_names[1]]
        self.val_metric_history = history.history.get(f'val_{self.model.metrics_names[1]}', [])
        
        
        if self.x_val is None:
            best_metric_idx = np.argmax(self.train_metric_history) if self.mode == 'max' else np.argmin(self.train_metric_history)
        else:
            best_metric_idx = np.argmax(self.val_metric_history) if self.mode == 'max' else np.argmin(self.val_metric_history)
            
        
        self.best_model_stats = {
            'loss': self.train_loss_history[best_metric_idx],
            self.model.metrics_names[1]: self.train_metric_history[best_metric_idx]
        }
        print(self.model.metrics_names[1])

        if self.x_val is not None:
            self.best_model_stats.update({
                'val_loss': self.val_loss_history[best_metric_idx],
                f'val_{self.model.metrics_names[1]}': self.val_metric_history[best_metric_idx]
            })
        
        print(f"Best {self.target}: {self.best_model_stats[self.target]}")