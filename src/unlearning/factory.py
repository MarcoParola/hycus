from src.unlearning_methods.base import BaseUnlearningMethod
from src.unlearning_methods.scrub import Scrub
from src.unlearning_methods.badT import BadT
from src.unlearning_methods.ssd import SSD

def get_unlearning_method(method_name, model, retain_loader, forget_loader, test_loader, train_loader, cfg):
    methods = {
        'scrub': Scrub(cfg, model, retain_loader, forget_loader, test_loader),
        'badT': BadT(cfg, model, retain_loader, forget_loader),
        'ssd': SSD(cfg, model)
    }
    
    if method_name not in methods:
        raise ValueError(f"Unlearning method '{method_name}' not recognised.")
    
    return methods[method_name]
