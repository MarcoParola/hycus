import torch
import time, copy
import torch.nn as nn
import tqdm
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from src.unlearning_methods.base import BaseUnlearningMethod

class Scrub(BaseUnlearningMethod):

    def __init__(self, opt, model, retain_loader, forget_loader, test_loader, maximize=False, alpha=0.001, kd_T=4.0):
        super().__init__(opt, model)
        print("Inizializzazione di Scrub")
        self.og_model = copy.deepcopy(model)  # Copia del modello originale
        self.criterion = nn.CrossEntropyLoss()
        self.retain_loader = retain_loader
        self.forget_loader = forget_loader
        self.test_loader = test_loader
        self.maximize = maximize
        self.alpha = alpha
        self.kd_T = kd_T
        self.msteps = opt.train_iters // 2  
        self.save_files = {"train_time_taken": 0, "val_top1": []}
        self.curr_step = 0  

    def unlearn(self, train_loader):
        print("Inizio processo di unlearning")
        """Processo di unlearning con massimizzazione/minimizzazione alternata"""
        self.curr_step=0
        while self.curr_step < self.opt.train_iters:
            if self.curr_step < self.msteps:
                self.maximize = True
                self._train_one_phase(loader=self.forget_loader, train_loader=train_loader)
           
            self.curr_step += 1
            self.maximize = False
            self._train_one_phase(loader=train_loader, train_loader=train_loader)
            

    def distill_kl_loss(self, student_output, teacher_output, temperature):
        """Calcola la perdita di distillazione usando la KL-divergenza."""
        student_output = F.log_softmax(student_output / temperature, dim=1)
        teacher_output = F.softmax(teacher_output / temperature, dim=1)
        return F.kl_div(student_output, teacher_output, reduction='batchmean') * (temperature ** 2)

    def _train_one_phase(self, loader, train_loader):
        """Gestisce una fase di training, massimizzazione o minimizzazione."""
        time_start = time.process_time()
        self.train_one_epoch(loader=loader)
        self.save_files['train_time_taken'] += time.process_time() - time_start
        self.eval(self.test_loader)

    def forward_pass(self, inputs, target, infgt):
        """Esegue il forward pass e calcola la perdita."""
        
        inputs, target = inputs.to(self.opt.device), target.to(self.opt.device)        
        # Forward pass del modello attuale (con gradienti abilitati)
        output = self.model(inputs)
        # Forward pass del modello originale (senza gradienti)
        with torch.no_grad():
            logit_t = self.og_model(inputs)
        # Calcolo della perdita: standard (cross-entropy) + distillazione (KL-divergenza)
        loss = F.cross_entropy(output, target)
        loss += self.alpha * self.distill_kl_loss(output, logit_t, self.kd_T)
        # Se maximization è True, inverto la perdita
        if self.maximize:
            loss = -loss
        # Restituisco sia la perdita che le predizioni (output)
        return output, loss


    