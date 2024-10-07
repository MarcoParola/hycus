"""import torch
import time, copy
import torch.nn as nn
import tqdm
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from src.unlearning_methods.base import BaseUnlearningMethod

class Icus(BaseUnlearningMethod):

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
        
        while self.curr_step < self.opt.train_iters:
            if self.curr_step < self.msteps:
                self.maximize = True
                self._train_one_phase(loader=self.forget_loader)
                self.curr_step += 1
                self.eval(self.test_loader)
           
            self.maximize = False
            self._train_one_phase(loader=train_loader)
            self.curr_step += 1

    def distill_kl_loss(self, student_output, teacher_output, temperature):
        
        student_output = F.log_softmax(student_output / temperature, dim=1)
        teacher_output = F.softmax(teacher_output / temperature, dim=1)
        return F.kl_div(student_output, teacher_output, reduction='batchmean') * (temperature ** 2)

    def _train_one_phase(self, loader):
        
        time_start = time.process_time()
        self.train_one_epoch(loader=loader)
        self.save_files['train_time_taken'] += time.process_time() - time_start
        #self.eval(loader=self.test_loader)  
        self.eval(self.test_loader)

    def forward_pass(self, inputs, target, infgt):
    
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


    #FORWARD PASS
    # recupera pesi della classe target 
    weights = self.model.last_layer_weights(target)
    descr = self.description[target]
    weights_encoded = self.icus_weights_encoder(weights)
    descr_encoded = self.icus_descr_encoder(descr)
    # calcola la distanza tra i pesi e la descrizione
    loss=0
    dist = self.icus_distance(weights_encoded, descr_encoded) #cosise similarity (?)
    loss = dist
    descr_decoded = self.icus_descr_decoder(descr_encoded)
    loss += F.mse_loss(descr, descr_decoded)
    weights_decoded = self.icus_weights_decoder(weights_encoded)
    if infgt==0:
        loss += F.mse_loss(weights, weights_decoded)
    else:
        random_weights = self.model.last_layer_weights(random_target)
        loss += F.mse_loss(random_weights, weights_decoded)"""


#ROBA DI GPT DOPO CONTROLLA

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
from src.unlearning_methods.base import BaseUnlearningMethod

# Classe Icus
class Icus(BaseUnlearningMethod):
    def __init__(self, opt, model, input_dim, nclass):
        super(Icus, self).__init__(opt, model)
        self.fc = nn.Linear(input_dim, nclass)  # Layer finale per nclass classi
        # Inizializza le descrizioni e altri componenti necessari
        self.description = {}  # Riempi questo con le descrizioni
        self.icus_weights_encoder = nn.Linear(50, 50)  # Esempio di encoder
        self.icus_descr_encoder = nn.Linear(50, 50)  # Esempio di encoder
        self.icus_descr_decoder = nn.Linear(50, 50)  # Esempio di decoder
        self.icus_weights_decoder = nn.Linear(50, 50)  # Esempio di decoder
        self.current_step = 0

    def last_layer_weights(self, target):
        # Implementa la logica per recuperare i pesi della classe target
        return self.fc.weight[target]

    def icus_distance(self, weights_encoded, descr_encoded):
        # Calcola la distanza (ad esempio, similarità coseno)
        return nn.functional.cosine_similarity(weights_encoded, descr_encoded)

    #Ciclo di unlearning
    def unlearn(self, model, train_loader):
        model.train()  # Imposta il modello in modalità di addestramento
        self.current_step = 0
        while self.current_step < self.opt.train_iters:
            self.current_step += 1
            running_loss = 0.0
            """for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.opt.device), targets.to(self.opt.device)
                
                self.optimizer.zero_grad()  # Azzerare i gradienti
                
                # Calcola la perdita utilizzando il metodo forward
                # Nota: dovresti fornire un random_target e infgt a tuo piacere.
                loss = model(inputs, targets)  # Chiamata al metodo forward
                
                # Backpropagation
                loss.backward()
                self.optimizer.step()  # Aggiorna i pesi
                
                running_loss += loss.item()"""
            self.train_one_epoch(train_loader)
                
                if (i + 1) % 10 == 0:  # Stampa ogni 10 batch
                    print(f'Epoch [{self.current_step + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            
            # Stampa la perdita media per l'epoca
            print(f'Epoch [{self.curr_step}/{num_epochs}], Average Loss: {running_loss / len(train_loader):.4f}')

# pesi | descrizione | infgt | target 
    def train_one_epoch(self, loader):
        print("Inizio epoca di training")
        self.model.train()  # Imposta il modello in modalità training
        running_loss = 0.0


    def forward(self, x, target, random_target=None, infgt=0):
        # Recupera pesi della classe target 
        weights = self.last_layer_weights(target)
        descr = self.description[target]
        weights_encoded = self.icus_weights_encoder(weights)
        descr_encoded = self.icus_descr_encoder(descr)

        # Calcola la distanza tra i pesi e la descrizione
        loss = 0
        dist = self.icus_distance(weights_encoded, descr_encoded)
        loss = dist
        
        descr_decoded = self.icus_descr_decoder(descr_encoded)
        loss += nn.MSELoss()(descr, descr_decoded)

        weights_decoded = self.icus_weights_decoder(weights_encoded)
        if infgt == 0:
            loss += nn.MSELoss()(weights, weights_decoded)
        else:
            random_weights = self.last_layer_weights(random_target)
            loss += nn.MSELoss()(random_weights, weights_decoded)

        return loss  # Restituisci la perdita


# 4. Esecuzione dell'Addestramento
batch_size = 64
num_epochs = 10
train_loader = prepare_data(batch_size)
train(model, train_loader, optimizer, num_epochs)
