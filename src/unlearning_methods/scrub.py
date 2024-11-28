import torch
import time, copy
import torch.nn as nn
import tqdm
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from src.metrics.metrics import compute_metrics
from src.utils import LinearLR
from src.unlearning_methods.base import BaseUnlearningMethod

class Scrub(BaseUnlearningMethod):

    def __init__(self, opt, model, forgetting_subset, logger, alpha=0.1, kd_T=1.0):
        super().__init__(opt, model)
        print("Inizializzazione di Scrub")
        self.og_model = copy.deepcopy(model)  # original model copy
        self.forgetting_subset = forgetting_subset
        self.opt=opt
        self.criterion = nn.CrossEntropyLoss()
        self.logger = logger
        self.alpha = alpha
        self.kd_T = kd_T
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=opt.unlearn.lr, momentum=0.9, weight_decay=0.001)
        #self.scheduler = LinearLR(self.optimizer, T=self.opt.train_iters*1.25, warmup_epochs=self.opt.train_iters//100) # Spend 1% time in warmup, and stop 66% of the way through training 
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        self.msteps = opt.unlearn.scrub_steps//2 
        self.save_files = {"train_time_taken": 0, "val_top1": []}
        self.curr_step = 0  

    def unlearn(self, retain_loader, forget_loader, val_loader=None):
        print("Start unlearning process")
        if val_loader is not None:
            self.val_loader = val_loader
        self.curr_step = 0
        self.epoch = 0

        while self.epoch < self.opt.unlearn.scrub_steps:
            print(f"Epoch {self.epoch}")
            print(f"Msteps {self.msteps}")
            if self.epoch < self.msteps:
                self.maximize = True
                self._train_one_phase(loader=forget_loader)
            
            self.maximize = False
            self._train_one_phase(loader=retain_loader)
            if self.val_loader is not None:
                metrics = compute_metrics(self.model, val_loader, self.opt.dataset.classes, self.forgetting_subset)
                self.logger.log_metrics({'accuracy_retain': metrics['accuracy_retaining'], 'accuracy_forget': metrics['accuracy_forgetting']})
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
        self.epoch += 1


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

