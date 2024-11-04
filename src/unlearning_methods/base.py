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
        self.scaler = GradScaler()  # mixed precision
        self.save_files = {"train_time_taken": 0} 
        # optional prenet
        if prenet is not None:
            self.prenet = prenet.to(opt.device)
        else:
            self.prenet = None

    def unlearn(self, train_loader, test_loader, val_loader=None):
        self.epoch = 0
        while self.epoch < self.opt.max_epochs: 
            time_start = time.process_time() 
            self.train_one_epoch(loader=train_loader) 
            self.epoch += 1
            self.validate(val_loader) # Validate the model
            self.save_files['train_time_taken'] += time.process_time() - time_start 
        return self.model

    def _training_step(self, inputs, labels):
        """Single step of training."""
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
        """Calculate KL-divergence loss"""
        student_outputs = torch.nn.functional.log_softmax(student_outputs / temperature, dim=1)
        teacher_outputs = torch.nn.functional.softmax(teacher_outputs / temperature, dim=1)
        return torch.nn.functional.kl_div(student_outputs, teacher_outputs, reduction='batchmean') * (temperature ** 2)

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.opt.device))
        self.model.to(self.opt.device)

    def train_one_epoch(self, loader):
        print("New training epoch")
        self.model.train()  
        # For each batch in the loader
        for inputs, labels in tqdm.tqdm(loader):
            inputs, labels = inputs.to(self.opt.device), labels.to(self.opt.device)
            with autocast(): 
                # Zero the gradients
                self.optimizer.zero_grad()
                # Forward pass
                preds, loss = self.forward_pass(inputs, labels)
                self.logger.log_metrics({"method":self.opt.unlearning_method, "loss": loss.item()}, step=self.curr_step)
                self.scaler.scale(loss).backward()  # Backward pass
                self.scaler.step(self.optimizer) # Update the weights
                self.scaler.update() # Update the scaler
                self.scheduler.step() # Update the learning rate
                self.curr_step += 1
                if self.curr_step > self.opt.train_iters:
                    break
        return


    def eval(self, loader, save_model=True, save_preds=False):
        """Evaluate a model basing on difference between retain and forget accuracy"""
        self.model.eval()   
        self.top1 = -1  # Reset top1
        correct_retain=0
        correct_forget=0
        total_retain=0
        total_forget=0

        if save_preds:
            preds, targets = [], []  # Lists to store predictions and targets

        with torch.no_grad():  # Disable gradient calculation
            for (images, target) in tqdm.tqdm(loader):  
               
                images, target = images.to(self.opt.device), target.to(self.opt.device)  # GPU
                output = self.model(images) if self.prenet is None else self.model(self.prenet(images))  # Forward pass

                # Compute accuracy
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

                if save_preds: 
                    preds.append(output.cpu().numpy())
                    targets.append(target.cpu().numpy())

        top1 = (correct_retain/total_retain) - (correct_forget/total_forget)
        self.top1 = -1

        if not save_preds:
            print(f'Epoca: {self.epoch} Val Top1: {top1*100:.2f}%')

        if save_model:
            if top1 > self.best_top1:  # If the best found until now
                self.best_top1 = top1  
                self.best_model = copy.deepcopy(self.model).cpu()  
                print(f"Nuovo best model salvato con accuratezza: {top1*100:.2f}%")

        self.model.train()  

        if save_preds:  
            preds = np.concatenate(preds, axis=0)
            targets = np.concatenate(targets, axis=0)
            return preds, targets  
        return

    def validate(self, val_loader):
        print("Start validation")
        self.model.eval()  
        correct_forget = 0
        total_forget = 0
        correct_retain = 0
        total_retain = 0

        with torch.no_grad():  # Disable gradient calculation

            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.opt.device), targets.to(self.opt.device)
                outputs = self.model(inputs)
                for o, t in zip(outputs, targets):
                    pred = torch.argmax(o)
                    if pred in self.forgetting_subset:
                        if pred==t:
                            correct_forget+=1
                        total_forget+=1
                    else:
                        if pred==t:
                            correct_retain+=1
                    total_retain+=1
        
        accuracy_retain = correct_retain / total_retain
        accuracy_forget = correct_forget / total_forget

        # Logging on WandB
        self.logger.log_metrics({
            "validation_retain_accuracy": accuracy_retain,
            "validation_forget_accuracy": accuracy_forget,
            "step": self.curr_step if hasattr(self, "curr_step") else self.epoch
        })
        
        self.model.train()  
        
def get_unlearning_method(cfg, method_name, model, unlearning_train, forgetting_set, logger=None):
    from src.unlearning_methods.scrub import Scrub
    from src.unlearning_methods.badT import BadT
    from src.unlearning_methods.ssd import SSD
    from src.unlearning_methods.icus import Icus
    if method_name == 'scrub':
        return Scrub(cfg, model, forgetting_set, logger)
    elif method_name == 'badT':
        return BadT(cfg, model, forgetting_set, logger)
    elif method_name == 'ssd':
        return SSD(cfg, model, logger)
    elif method_name == 'icus':
        return Icus(cfg, model, 128, cfg.dataset.classes, unlearning_train, forgetting_set, logger)
    else:
        raise ValueError(f"Unlearning method '{method_name}' not recognised.")
    return None