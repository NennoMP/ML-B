import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator
from keras.losses import BinaryCrossentropy
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold                 
        
from training.callbacks import CustomPrintCallback, CustomBestEarlyStopping

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
    
    def __init__(self, model, train_data, train_labels, val_data, val_labels, **kwargs):
        """
        Construct a new Solver instance.

        Args:
        - model: a model object

        - train_data: training data
        - train_labels: training gtc labels
        
        - val_data: validation data
        - val_labels: validation gtc labels
        """
        
        self.model = model
        self.train_data = train_data
        self.train_labels = train_labels
        self.val_data = val_data
        self.val_labels = val_labels
        self._reset()
    
    def _reset(self):
        """
        Reset/Setup some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.best_model_stats = None
        self.best_params = None

        self.train_loss_history = []
        self.val_loss_history = []
        
        # Record accuracy history
        self.train_accuracy_history = []
        self.val_accuracy_history = []
        
    def plot_history(self, out_path: str):
        epochs = list(range(1, len(self.train_loss_history) + 1))

        plt.figure(figsize=(12, 5))

        # Plotting BCE losses
        ax1 = plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot = Losses
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.plot(epochs, self.train_loss_history, label='Training', marker='o')
        ax1.plot(epochs, self.val_loss_history, label='Validation', marker='o')
        ax1.set_title('BCE losses')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Plotting accuracy
        ax2 = plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot = accuracy
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.plot(epochs, self.train_accuracy_history, label='Trainining', marker='o')
        ax2.plot(epochs, self.val_accuracy_history, label='Validation', marker='o')
        #ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
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
        """
        
        callbacks = [CustomPrintCallback(epochs=epochs)]
        
        if patience:
            early_stopping = CustomBestEarlyStopping(monitor='val_accuracy', patience=patience, mode='max', verbose=1, restore_best_weights=True)
            callbacks.append(early_stopping)

        history = self.model.fit(self.train_data, self.train_labels,
                                 validation_data=(self.val_data, self.val_labels),
                                 epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
        
        # Record loss history
        self.train_loss_history = history.history['loss']
        self.val_loss_history = history.history['val_loss']
        
        # Record accuracy history
        self.train_accuracy_history = history.history['accuracy']
        self.val_accuracy_history = history.history['val_accuracy']
        
        best_accuracy_idx = np.argmax(self.val_accuracy_history)
        self.best_model_stats = {'val_loss': self.val_loss_history[best_accuracy_idx],
                                 'train_loss': self.train_loss_history[best_accuracy_idx],
                                 'val_accuracy': self.val_accuracy_history[best_accuracy_idx],
                                 'accuracy': self.train_accuracy_history[best_accuracy_idx]}
        
        print(f"Best validation accuracy: {self.best_model_stats['val_accuracy']}")