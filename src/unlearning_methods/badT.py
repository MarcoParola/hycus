import torch, copy
import torch.nn.functional as F
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from src.unlearning_methods.base import BaseUnlearningMethod

class BadT(BaseUnlearningMethod):
    def __init__(self, opt, model, retain_loader, forget_loader):
        super().__init__(opt, model)
        self.og_model = copy.deepcopy(model)  # Copio il modello originale
        self.og_model.to(self.opt.device)
        self.og_model.eval()  # Metto il modello originale in modalità di valutazione
        
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
        # Passaggio forward senza prenet
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
        
        # Calcolo la top1 accuracy
        self.top1(output, target)
        
        return loss

    # Metodo train_one_epoch
    def train_one_epoch(self, loader):
        self.model.train()  # Imposto il modello in modalità di training
        for input, target, infgt in loader:  # Presumo che `infgt` sia parte del dataloader
            input, target, infgt = input.cuda(), target.cuda(), infgt.cuda()
            with autocast():  # Usando mixed precision per accelerare il training
                self.optimizer.zero_grad()
                loss = self.forward_pass(input, target, infgt)  # Richiamo il forward pass
                self.scaler.scale(loss).backward()  # Backprop con scaling della perdita
                self.scaler.step(self.optimizer)  # Aggiorna i pesi
                self.scaler.update()  # Aggiorna lo scaler per la mixed precision
            self.curr_step += 1
            if self.curr_step >= self.opt.train_iters:
                break
