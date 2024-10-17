from src.unlearning_methods.base import BaseUnlearningMethod
from src.unlearning_methods.scrub import Scrub
from src.unlearning_methods.badT import BadT
from src.unlearning_methods.ssd import SSD

def get_unlearning_method(method_name, model, retain_loader, forget_loader, test_loader, train_loader,wrapped_val_loader, cfg, forgetting_set, logger=None):
    if method_name == 'scrub':
        return Scrub(cfg, model, retain_loader, forget_loader, test_loader, wrapped_val_loader, logger)
    elif method_name == 'badT':
        return BadT(cfg, model, retain_loader, forget_loader, forgetting_set, logger)
    elif method_name == 'ssd':
        return SSD(cfg, model, logger)
    else:
        raise ValueError(f"Unlearning method '{method_name}' not recognised.")
    return -1