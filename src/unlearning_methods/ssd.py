import torch, copy, time
import torch.nn as nn
from src.utils import ssd_tuning
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from torch.nn import functional as F
from src.unlearning_methods.base import BaseUnlearningMethod
import tqdm


class SSD(BaseUnlearningMethod):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)

    def unlearn(self, wrapped_train_loader, test_loader, forget_loader):
        actual_iters = self.opt.train_iters
        self.opt.train_iters = len(wrapped_train_loader) + len(forget_loader)
        time_start = time.process_time()
        # Call the SSD tuning method to modify the model
        self.best_model = ssd_tuning(self.model, forget_loader, self.opt.SSDdampening, self.opt.SSDselectwt, wrapped_train_loader, self.opt.device)
        self.save_files['train_time_taken'] += time.process_time() - time_start
        self.opt.train_iters = actual_iters
        return

    def get_save_prefix(self):
        self.unlearn_file_prefix = f"{self.opt.pretrain_file_prefix}/{self.opt.deletion_size}_{self.opt.unlearn_method}_{self.opt.exp_name}"
        self.unlearn_file_prefix += f"_{self.opt.train_iters}_{self.opt.k}_{self.opt.SSDdampening}_{self.opt.SSDselectwt}"
        return self.unlearn_file_prefix

"""
import torch
import time
from torch.cuda.amp import autocast, GradScaler
from src.unlearning_methods.base import BaseUnlearningMethod
from utils import ssd_tuning

class SSD(BaseUnlearningMethod):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)
        self.scaler = GradScaler()  # Aggiunto scaler se utilizzi mixed precision

    def unlearn(self, wrapped_train_loader, test_loader, forget_loader):
        self._prepare_unlearning(wrapped_train_loader, forget_loader)
        
        time_start = time.process_time()

        # Call SSD tuning
        self.best_model = ssd_tuning(
            self.model, forget_loader, 
            self.opt.SSDdampening, self.opt.SSDselectwt, 
            wrapped_train_loader, self.opt.device
        )
        
        self._log_time("train_time_taken", time_start)

    def _prepare_unlearning(self, wrapped_train_loader, forget_loader):
        self.opt.train_iters = len(wrapped_train_loader) + len(forget_loader)
    
    def _log_time(self, log_key, time_start):
        
        self.save_files[log_key] += time.process_time() - time_start

    def get_save_prefix(self):
        
        return (f"{self.opt.pretrain_file_prefix}/{self.opt.deletion_size}_"
                f"{self.opt.unlearn_method}_{self.opt.exp_name}_"
                f"{self.opt.train_iters}_{self.opt.k}_"
                f"{self.opt.SSDdampening}_{self.opt.SSDselectwt}")
"""