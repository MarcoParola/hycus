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

    def __init__(self, opt, model, retain_loader, forget_loader, test_loader, maximize=False, alpha=1.0, kd_T=1.0):
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
        self.curr_step = 0  

    def unlearn(self, train_loader):
        print("Inizio processo di unlearning")
        """Processo di unlearning con massimizzazione/minimizzazione alternata"""
        while self.curr_step < self.opt.train_iters:
            if self.curr_step < self.msteps:
                self.maximize = True
                self._train_one_phase(loader=self.forget_loader, train_loader=train_loader)
            else:
                self.maximize = False
                self._train_one_phase(loader=train_loader, train_loader=train_loader)
            self.curr_step += 1

    def _train_one_phase(self, loader, train_loader):
        """Gestisce una fase di training, massimizzazione o minimizzazione."""
        time_start = time.process_time()
        self.train_one_epoch(loader=loader)
        self.save_files['train_time_taken'] += time.process_time() - time_start
        #self.eval(loader=self.test_loader)  
        self.eval(self.test_loader)

    def forward_pass(self, inputs, target):
        """Esegue il forward pass e calcola la perdita."""
        inputs, target = inputs.to(self.opt.device), target.to(self.opt.device)
        output = self.model(inputs)

        # Forward pass del modello originale (in modalità no_grad)
        with torch.no_grad():
            logit_t = self.og_model(inputs)

        # Perdita standard (cross-entropy) + distillazione (KL-divergenza)
        loss = F.cross_entropy(output, target)
        loss += self.alpha * self.distill_kl_loss(output, logit_t, self.kd_T)

        # Se maximization è True, inverto la perdita
        if self.maximize:
            loss = -loss

        # Calcolo accuratezza top-1
        self.top1(output, target)
        return loss

    def train_one_epoch(self, loader):
        """Esegue un'epoca di training su un loader."""
        print("Un passo di training")
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.opt.train.lr, momentum=0.9, weight_decay=0.001)
        running_loss = 0.0

        for inputs, labels in loader:
            print("Un batch")
            inputs, labels = inputs.to(self.opt.device), labels.to(self.opt.device)

            # Azzeramento gradienti
            optimizer.zero_grad()

            print("Forward pass e backward pass")
            loss = self.forward_pass(inputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        return running_loss / len(loader)

    def distill_kl_loss(self, student_output, teacher_output, temperature: float):
        """Calcola la perdita di distillazione usando la KL-divergenza."""
        student_output = F.log_softmax(student_output / temperature, dim=1)
        teacher_output = F.softmax(teacher_output / temperature, dim=1)
        return F.kl_div(student_output, teacher_output, reduction='batchmean') * (temperature ** 2)

    
    def eval(self, loader, save_model=True, save_preds=False):
        """Valuta il modello su un dataset e salva il best model basato sulla top-1 accuracy."""
        self.model.eval()   # Imposta il modello in modalità di valutazione
        self.top1.reset()   # Resetta il calcolo dell'accuracy

        if save_preds:
            preds, targets = [], []  # Liste per salvare predizioni e target

        with torch.no_grad():  # Disabilita il calcolo dei gradienti durante la valutazione
            for (images, target) in tqdm.tqdm(loader):  # Per ogni batch nel loader di test/validazione
               
                images, target = images.to(self.opt.device), target.to(self.opt.device)  # Sposta su GPU
                output = self.model(images) if self.prenet is None else self.model(self.prenet(images))  # Forward pass

                # Calcola l'accuracy
                self.top1.update(output, target)

                if save_preds:  # Salva le predizioni e i target
                    preds.append(output.cpu().numpy())
                    targets.append(target.cpu().numpy())

        # Calcola l'accuratezza top-1
        top1 = self.top1.compute().item()  
        self.top1.reset()

        # Stampa l'accuratezza
        if not save_preds:
            print(f'Step: {self.curr_step} Val Top1: {top1*100:.2f}%')

        # Se è abilitato il salvataggio del modello
        if save_model:
            self.save_files['val_top1'].append(top1)  # Salva l'accuratezza nel log
            if top1 > self.best_top1:  # Se è la migliore accuratezza finora
                self.best_top1 = top1  # Aggiorna la best_top1
                self.best_model = copy.deepcopy(self.model).cpu()  # Salva il modello migliore
                print(f"Nuovo best model salvato con accuratezza: {top1*100:.2f}%")

        self.model.train()  # Torna in modalità di training

        if save_preds:  # Se devi salvare predizioni e target
            preds = np.concatenate(preds, axis=0)
            targets = np.concatenate(targets, axis=0)
            return preds, targets  # Ritorna predizioni e target
        return
