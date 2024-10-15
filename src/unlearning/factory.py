from src.unlearning_methods.base import BaseUnlearningMethod
from src.unlearning_methods.scrub import Scrub
from src.unlearning_methods.badT import BadT
from src.unlearning_methods.ssd import SSD

def get_unlearning_method(method_name, model, retain_loader, forget_loader, test_loader, train_loader, cfg, forgetting_set):
    if method_name == 'scrub':
        return Scrub(cfg, model, retain_loader, forget_loader, test_loader, forgetting_set)
    elif method_name == 'badT':
        return BadT(cfg, model, retain_loader, forget_loader, forgetting_set)
    elif method_name == 'ssd':
        return SSD(cfg, model)
    else:
        raise ValueError(f"Unlearning method '{method_name}' not recognised.")
    return -1