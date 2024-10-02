import torch
from abc import ABC, abstractmethod
from torch.cuda.amp import GradScaler
import torchmetrics

class BaseUnlearningMethod(ABC):
    def __init__(self, opt, model, prenet=None):
        self.opt = opt
        self.model = model.to(opt.device)
        self.best_top1 = 0
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
        self.top1 = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.scaler = GradScaler()  # Aggiunto per supportare mixed precision
        self.save_files = {"train_time_taken": 0}  # Inizializzazione log
        
        # Prenet opzionale
        if prenet is not None:
            self.prenet = prenet.to(opt.device)
        else:
            self.prenet = None


    def _training_step(self, inputs, labels):
        """Esegue un singolo step di training con supporto per mixed precision."""
        inputs, labels = inputs.to(self.opt.device), labels.to(self.opt.device)
        
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
        print("Inizio epoca di training")
        self.model.train()  # Imposta il modello in modalità training
        self.top1.reset()  # Reset della metrica all'inizio dell'epoca
        running_loss = 0.0
        self.curr_step=0
        # Ciclo principale su ogni batch del loader
        for inputs, labels, infgt in loader:
            print("Un batch")
            inputs, labels, infgt = inputs.to(self.opt.device), labels.to(self.opt.device), infgt.to(self.opt.device)
            # Azzeramento dei gradienti
            self.optimizer.zero_grad()
            # Eseguiamo il forward pass e calcoliamo la perdita
            preds, loss = self.forward_pass(inputs, labels, infgt)
            # Backward pass e aggiornamento pesi
            loss.backward()
            self.optimizer.step()
            # Aggiornamento della metrica per il batch corrente
            self.top1.update(preds, labels)
            # Accumula la perdita
            running_loss += loss.item()
            # Incremento del contatore step
            self.curr_step += 1
            # Facoltativo: interrompe se si raggiunge il numero massimo di step
            if self.curr_step >= self.opt.train_iters:
                break
        # Calcolo dell'accuratezza dopo aver processato tutti i batch
        top1 = self.top1.compute().item()
        self.top1.reset()  # Reset della metrica per la prossima epoca
        print(f'Step: {self.curr_step} Train Top1: {top1:.3f}, Loss: {running_loss:.4f}')
        return