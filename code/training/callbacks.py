from keras.callbacks import EarlyStopping


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