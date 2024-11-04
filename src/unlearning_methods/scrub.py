import torch
import time, copy
import torch.nn as nn
import tqdm
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from src.utils import LinearLR
from src.unlearning_methods.base import BaseUnlearningMethod

class Scrub(BaseUnlearningMethod):

    def __init__(self, opt, model, forgetting_subset, logger, alpha=0.1, kd_T=4.0):
        super().__init__(opt, model)
        print("Inizializzazione di Scrub")
        self.og_model = copy.deepcopy(model)  # original model copy
        self.forgetting_subset = forgetting_subset
        self.criterion = nn.CrossEntropyLoss()
        self.logger = logger
        self.alpha = alpha
        self.kd_T = kd_T
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-6, momentum=0.9, weight_decay=0.9)
        self.scheduler = LinearLR(self.optimizer, T=self.opt.train_iters*1.25, warmup_epochs=self.opt.train_iters//100) # Spend 1% time in warmup, and stop 66% of the way through training 
        self.msteps = 20000 # Misleading value, to be changed. Probably the logic in changing curr_step should be changed. I think
                            # it should be updated after each epoch, not after each step, otherwise there's the risk of stopping
                            # in the middle of an epoch where loss is too high.
                            # I set msteps to a so high value to make it ininfluent. TO REVIEW
        self.save_files = {"train_time_taken": 0, "val_top1": []}
        self.curr_step = 0  

    def unlearn(self, retain_loader, forget_loader, val_loader=None):
        print("Start unlearning process")
        if val_loader is not None:
            self.val_loader = val_loader

        while self.curr_step < self.opt.train_iters:
            if self.curr_step < self.msteps:
                self.maximize = True
                self._train_one_phase(loader=forget_loader)
            
            self.maximize = False
            self._train_one_phase(loader=retain_loader)
            if self.val_loader is not None:
                self.validate(self.val_loader)
        return self.model
            

    def distill_kl_loss(self, student_output, teacher_output, temperature):
        """Calculate the KL-divergence loss for knowledge distillation."""
        student_output = F.log_softmax(student_output / temperature, dim=1)
        teacher_output = F.softmax(teacher_output / temperature, dim=1)
        loss = F.kl_div(student_output, teacher_output, reduction='sum')
        loss = loss * (temperature ** 2) / student_output.shape[0]
        return loss

    def _train_one_phase(self, loader):
        time_start = time.process_time()
        self.train_one_epoch(loader=loader)
        self.save_files['train_time_taken'] += time.process_time() - time_start

    def forward_pass(self, inputs, target):
        inputs, target = inputs.to(self.opt.device), target.to(self.opt.device)        
        # Forward pass (with gradients)
        output = self.model(inputs)
        # Forward pass (without gradients)
        with torch.no_grad():
            logit_t = self.og_model(inputs)
        # Calculate loss: standard (cross-entropy) + distillation (KL-divergence)
        loss = F.cross_entropy(output, target)
        loss += self.alpha * self.distill_kl_loss(output, logit_t, self.kd_T)
        if self.maximize:
            loss = -loss
        
        return output, loss


