import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import omegaconf

def get_save_model_callback(save_path):
    """Returns a ModelCheckpoint callback
    
    save_path: str: save path
    """
    save_model_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=save_path,
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        save_last=True,
    )
    return save_model_callback

def get_early_stopping(patience=10):
    """Returns an EarlyStopping callback

    patience: int: patience
    """
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=patience,
    )
    return early_stopping_callback

# forgetting_set could be a list or a string
def get_forgetting_subset(forgetting_set, n_classes, forgetting_set_size):
    """Returns a forgetting subset

    forgetting_set: str or list: forgetting set
    n_classes: int: number of classes
    forgetting_set_size: int: forgetting set size
    """
    
    if forgetting_set == 'random':
        # return random values using torch
        return torch.randint(0, n_classes, (forgetting_set_size,)).tolist()
    elif isinstance(forgetting_set, omegaconf.listconfig.ListConfig):
        forgetting_set = list(forgetting_set)
        return forgetting_set
    else:
        print('Unknown forgetting set')
    return None
    
        


