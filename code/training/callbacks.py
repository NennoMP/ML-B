from keras.callbacks import EarlyStopping, Callback


class CustomPrintCallback(Callback):
    """Custom Callback printing useful metrics-related information for each epoch."""
    
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
    
    def on_epoch_end(self, epoch, logs=None):
        if 'val_accuracy' in logs:
            print(f"\nEpoch {epoch+1}/{self.epochs} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}\n")
        else:
            print(f"\nEpoch {epoch+1}/{self.epochs} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f}\n")


class CustomBestEarlyStopping(EarlyStopping):
    """Custom EarlyStopping to restore best model weights on train end."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def on_train_end(self, logs=None):
        if self.patience is None or self.patience < 0:
            return
        
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"\nEpoch {self.stopped_epoch + 1}: early stopping.")
        elif self.restore_best_weights:
            if self.verbose > 0:
                print("Restoring best model weights.")
            self.model.set_weights(self.best_weights)