import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator
from keras.losses import BinaryCrossentropy
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold                 
        
from training.callbacks import CustomPrintCallback, CustomBestEarlyStopping


MODE_MAPPING = {'loss': 'min', 'mean_euclidean_error': 'min', 'accuracy': 'max'}


class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training a classification/regression model.
    It performs gradient descent using the given learning rate.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    After training (i.e. when train() method returns), the best performing weights are loaded back into
    the model, and the best achieved stats are stored in best_model_stats. History records for train loss,
    val loss, train accuracy, and val accuracy are also memorized.
    """
    
    def __init__(self, model: any, x_train: np.ndarray, y_train: np.ndarray, x_val=None, y_val=None, target='accuracy', **kwargs):
        """
        Construct a new Solver instance.

        Args:
        - model: a model object

        - x_train: training data
        - y_train: training gtc labels
        
        - x_val: validation data
        - y_val: validation gtc labels
        """
        
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.validation_data = None
        
        
        if target not in MODE_MAPPING:
            raise ValueError(
                "Unknown <target> parameter, can only be 'loss', 'accuracy', or 'mean_euclidean_error'!"
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
        
    def plot_history(self, out_path: str):
        epochs = list(range(3, len(self.train_loss_history) + 1))

        plt.figure(figsize=(12, 5))

        # Plotting MSE losses
        ax1 = plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot = Losses
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.plot(epochs, self.train_loss_history[2:], label='Training', linestyle='--')
        ax1.plot(epochs, self.val_loss_history[2:], label='Internal test', linestyle='--')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE')
        ax1.legend()
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Plotting metric (MEE)
        ax2 = plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot = metric
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.plot(epochs, self.train_metric_history[2:], label='Training', linestyle='--')
        ax2.plot(epochs, self.val_metric_history[2:], label='Internal test', linestyle='--')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MEE')
        ax2.legend()
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.show()

    def train(self, epochs=50, batch_size=32, patience=None):
        """
        Run optimization to train the model.
        
        Args:
            epochs: number of epochs for training
            batch_size: size of the batch
            
        Optional:
            patience: nummber of epochs to wait for improvement before stopping
        """
    
            
        callbacks = []
        
        if patience:
            monitor = self.target if self.x_val is None else f'val_{self.target}'
            early_stopping = CustomBestEarlyStopping(monitor=monitor, patience=patience, mode=self.mode, verbose=1, restore_best_weights=True)
            callbacks.append(early_stopping)

        history = self.model.fit(self.x_train, self.y_train,
                                 validation_data=self.validation_data,
                                 epochs=epochs, batch_size=batch_size, callbacks=callbacks, shuffle=True, verbose=1)
        
        # Record loss history
        self.train_loss_history = history.history['loss']
        self.val_loss_history = history.history.get('val_loss', [])
        
        # Record metric history
        self.train_metric_history = history.history[self.target]
        self.val_metric_history = history.history.get(f'val_{self.target}', [])
        
        
        if self.x_val is None:
            best_metric_idx = np.argmax(self.train_metric_history) if self.mode == 'max' else np.argmin(self.train_metric_history)
        else:
            best_metric_idx = np.argmax(self.val_metric_history) if self.mode == 'max' else np.argmin(self.val_metric_history)
            
        
        self.best_model_stats = {
            'loss': self.train_loss_history[best_metric_idx],
            self.target: self.train_metric_history[best_metric_idx]
        }

        if self.x_val is not None:
            self.best_model_stats.update({
                'val_loss': self.val_loss_history[best_metric_idx],
                f'val_{self.target}': self.val_metric_history[best_metric_idx]
            })
        
        if self.x_val is not None:
            print(f"Best validation {self.target}: {self.best_model_stats[f'val_{self.target}']}")
        else:
            print(f"Best {self.target}: {self.best_model_stats[self.target]}")