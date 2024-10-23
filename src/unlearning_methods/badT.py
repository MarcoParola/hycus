import torch, copy
import torch.nn.functional as F
import torch.nn as nn
import tqdm
from torch.cuda.amp import autocast, GradScaler
from src.unlearning_methods.base import BaseUnlearningMethod

class BadT(BaseUnlearningMethod):
    def __init__(self, opt, model, forgetting_set, logger=None):
        super().__init__(opt, model)
        print("Inizializzazione di BadT")
        self.og_model = copy.deepcopy(model)  # Copio il modello originale
        self.og_model.to(self.opt.device)
        self.og_model.eval()  # Metto il modello originale in modalità di valutazione
        self.forgetting_set = forgetting_set
        self.logger=logger
        
        # Creo il modello random con pesi casuali
        self.random_model = copy.deepcopy(model)
        self.random_model.apply(self._random_weights_init)
        self.random_model.to(self.opt.device)
        self.random_model.eval()
        
        # Inizializzazione dell'ottimizzatore e scaler per mixed precision
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opt.train.lr, momentum=0.9, weight_decay=0.001)
        self.scaler = GradScaler()
        
        self.kltemp = 1  # Temperatura per la KL-divergenza (knowledge distillation)

    def _random_weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward_pass(self, sample, target, infgt):
        output = self.model(sample)
        
        # Calcolo dei logit dei due modelli (originale e random)
        full_teacher_logits = self.og_model(sample)
        unlearn_teacher_logits = self.random_model(sample)
        
        # Applico softmax con temperatura ai logit
        f_teacher_out = torch.nn.functional.softmax(full_teacher_logits / self.kltemp, dim=1)
        u_teacher_out = torch.nn.functional.softmax(unlearn_teacher_logits / self.kltemp, dim=1)
        
        # Seleziono quale output del modello usare in base a infgt
        labels = torch.unsqueeze(infgt, 1)
        overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
        
        # Calcolo la perdita con la KL-divergenza
        student_out = F.log_softmax(output / self.kltemp, dim=1)
        loss = F.kl_div(student_out, overall_teacher_out, reduction='batchmean')
        
        return output,loss



    def train_one_epoch(self, loader):
            """Esegue un'epoca di training su un loader."""
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
                    self.logger.log_metrics({"method":"BadT", "loss": loss.item()}, step=self.epoch)
                    #preds = torch.argmax(preds, dim=1)
                    self.scaler.scale(loss).backward()  #calcolo i gradienti e applico la backpropagation
                    self.scaler.step(self.optimizer) #aggiorno i pesi
                    self.scaler.update() #aggiorno lo scaler
                    self.scheduler.step() #aggiorno il learning rate
            print(f'Epoca: {self.epoch}')
            return