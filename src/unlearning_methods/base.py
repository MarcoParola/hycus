import torch
from abc import ABC, abstractmethod
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import numpy as np
import torchmetrics
import copy
import tqdm
import time
from src.utils import LinearLR

class BaseUnlearningMethod(ABC):
    def __init__(self, opt, model, forgetting_set=None, prenet=None):
        self.opt = opt
        self.model = model.to(opt.device)
        #self.save_files = {'train_top1':[], 'val_top1':[], 'train_time_taken':0}
        self.best_top1 = -1
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0025)
        self.scheduler = LinearLR(self.optimizer, T=self.opt.train_iters*1.25, warmup_epochs=self.opt.train_iters//100) # Spend 1% time in warmup, and stop 66% of the way through training 
        self.top1 = -1
        self.scaler = GradScaler()  # Aggiunto per supportare mixed precision
        self.save_files = {"train_time_taken": 0}  # Inizializzazione log
        # Prenet opzionale
        if prenet is not None:
            self.prenet = prenet.to(opt.device)
        else:
            self.prenet = None

    def unlearn(self, train_loader, test_loader, val_loader=None):
        self.epoch = 0
        while self.epoch < self.opt.max_epochs: #finchè non ho finito di trainare
            #self.curr_step = 0
            time_start = time.process_time() #salvo il tempo di inizio
            self.train_one_epoch(loader=train_loader) #traino per un'epoca
            self.epoch += 1
            self.validate(val_loader) #valuto
            self.save_files['train_time_taken'] += time.process_time() - time_start #salvo il tempo impiegato
        return

    def _training_step(self, inputs, labels):
        """Esegue un singolo step di training con supporto per mixed precision."""
        inputs, labels = inputs.to(self.opt.device), labels.to(self.opt.device)
        print(f"Inputs are on device: {inputs.device}, Labels are on device: {labels.device}")
        
        with torch.cuda.amp.autocast():
            outputs = self.model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        return loss.item()

    def compute_kl_loss(self, student_outputs, teacher_outputs, temperature):
        """Calcola la KL-divergence loss per distillazione del knowledge."""
        student_outputs = torch.nn.functional.log_softmax(student_outputs / temperature, dim=1)
        teacher_outputs = torch.nn.functional.softmax(teacher_outputs / temperature, dim=1)
        return torch.nn.functional.kl_div(student_outputs, teacher_outputs, reduction='batchmean') * (temperature ** 2)

    def save_model(self, filepath):
        """Salva il modello nel filepath specificato."""
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        """Carica il modello dal filepath specificato."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.opt.device))
        self.model.to(self.opt.device)

    def train_one_epoch(self, loader):
        """Esegue un'epoca di training su un loader."""
        self.epoch += 1
        print("Inizio epoca di training")
        self.model.train()  # Imposta il modello in modalità training

        # Ciclo principale su ogni batch del loader
        for inputs, labels, infgt in tqdm.tqdm(loader):
            inputs, labels, infgt = inputs.to(self.opt.device), labels.to(self.opt.device), infgt.to(self.opt.device)
            with autocast(): 
                # Azzeramento dei gradienti
                self.optimizer.zero_grad()
                # Eseguiamo il forward pass e calcoliamo la perdita
                preds, loss = self.forward_pass(inputs, labels, infgt)
                self.logger.log_metrics({"method":self.opt.unlearning_method, "loss": loss.item()}, step=self.curr_step)
                #preds = torch.argmax(preds, dim=1)
                self.scaler.scale(loss).backward()  #calcolo i gradienti e applico la backpropagation
                self.scaler.step(self.optimizer) #aggiorno i pesi
                self.scaler.update() #aggiorno lo scaler
                self.scheduler.step() #aggiorno il learning rate
                self.curr_step += 1
                if self.curr_step > self.opt.train_iters:
                    break
                
        print(f'Epoca: {self.epoch}')
        return


    def eval(self, loader, save_model=True, save_preds=False):
        """Valuta il modello su un dataset e salva il best model basato sulla top-1 accuracy."""
        self.model.eval()   # Imposta il modello in modalità di valutazione
        self.top1 = -1   # Resetta il calcolo dell'accuracy
        correct_retain=0
        correct_forget=0
        total_retain=0
        total_forget=0

        if save_preds:
            preds, targets = [], []  # Liste per salvare predizioni e target

        with torch.no_grad():  # Disabilita il calcolo dei gradienti durante la valutazione
            for (images, target) in tqdm.tqdm(loader):  # Per ogni batch nel loader di test/validazione
               
                images, target = images.to(self.opt.device), target.to(self.opt.device)  # Sposta su GPU
                output = self.model(images) if self.prenet is None else self.model(self.prenet(images))  # Forward pass

                # Calcola l'accuracy
                # self.top1.update(output, target)
                _, preds = torch.max(output, 1)
                for t, p in zip(target, preds):
                    if t in self.forgetting_set:
                        if p==t:
                            correct_forget+=1
                        total_forget+=1
                    else:
                        if p==t:
                            correct_retain+=1
                        total_retain+=1

                if save_preds:  # Salva le predizioni e i target
                    preds.append(output.cpu().numpy())
                    targets.append(target.cpu().numpy())

        # Calcola l'accuratezza top-1
        top1 = (correct_retain/total_retain) - (correct_forget/total_forget)
        self.top1 = -1

        # Stampa l'accuratezza
        if not save_preds:
            print(f'Epoca: {self.epoch} Val Top1: {top1*100:.2f}%')

        # Se è abilitato il salvataggio del modello
        if save_model:
            #self.save_files['val_top1'].append(top1)  # Salva l'accuratezza nel log
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

    def validate(self, val_loader):
        print("Start validation")
        self.model.eval()  # Imposta il modello in modalità valutazione
        correct_forget = 0
        total_forget = 0
        correct_retain = 0
        total_retain = 0

        with torch.no_grad():  # Disattiva il calcolo dei gradienti durante la validazione

            for inputs, targets, infgt in val_loader:
                inputs, targets, infgt = inputs.to(self.opt.device), targets.to(self.opt.device), infgt.to(self.opt.device)
                outputs = self.model(inputs)
                for i, o, t in zip(infgt, outputs, targets):
                    pred = torch.argmax(o)
                    print(f"Predizione: {pred}, Target: {t}", "Forget" if i == 1 else "Retain")
                    if i == 1:
                        if pred==t:
                            correct_forget+=1
                        total_forget+=1
                    else:
                        if pred==t:
                            correct_retain+=1
                    total_retain+=1
        
        accuracy_retain = correct_retain / total_retain
        accuracy_forget = correct_forget / total_forget

        # Puoi anche loggare i risultati della validazione, ad esempio su Weights & Biases
        self.logger.log_metrics({
            "validation_retain_accuracy": accuracy_retain,
            "validation_forget_accuracy": accuracy_forget,
            "step": self.curr_step if hasattr(self, "curr_step") else self.epoch
        })
        
        self.model.train()  # Torna in modalità training per il prossimo ciclo
