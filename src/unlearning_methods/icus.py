import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
from src.architectures.autoencoder import Autoencoder
from src.architectures.autoencoder import JointAutoencoder
from src.unlearning_methods.base import BaseUnlearningMethod
from src.utils import retrieve_weights


class Icus(BaseUnlearningMethod):
    def __init__(self, opt, model, input_dim, nclass, wrapped_train_loader, forgetting_subset):
        super().__init__(opt, model)
        self.fc = model.fc
        # Inizializza le descrizioni e altri componenti necessari
        self.description = wrapped_train_loader.dataset.descr
        flatten_description = self.description.view(nclass, -1)
        self.forgetting_subset = forgetting_subset
        weights, bias = retrieve_weights(model)
        # Concatenazione dei pesi e bias, assicurati che il bias abbia forma corretta
        self.weights = torch.cat((weights, bias.view(-1, 1)), dim=1)
        # Ottieni il numero di input features dai pesi
        input_features = self.weights.shape[1]  
        # Definizione degli autoencoder
        descr_ae = Autoencoder(opt, flatten_description.shape[1], embed_dim=512, num_layers=2)  
        weights_ae = Autoencoder(opt, input_features, embed_dim=512, num_layers=2) 
        # Joint Autoencoder
        self.joint_ae = JointAutoencoder(descr_ae, weights_ae)
        self.current_step = 0
        # Ottimizzatori per gli autoencoder
        self.descr_optimizer = optim.Adam(self.joint_ae.ae1.parameters(), lr=0.001)
        self.weights_optimizer = optim.Adam(self.joint_ae.ae2.parameters(), lr=0.001)
        

    def last_layer_weights(self, target):
        return self.fc.weight[target]

    def icus_distance(self, weights_encoded, descr_encoded):
        return nn.functional.cosine_similarity(weights_encoded, descr_encoded)

    # Ciclo di unlearning
    def unlearn(self, model, train_loader):
        model.train()  # Imposta il modello in modalità di addestramento
        self.current_step = 0
        for epoch in range(self.opt.epochs):
            print("Epoca: ", epoch)
            self.train_one_epoch(train_loader)

    def train_one_epoch(self, loader):
        print("Inizio epoca di training")
        self.joint_ae.train()  # Imposta il Joint Autoencoder in modalità training
        running_loss = 0.0

        for batch in loader:
            # Estrai il batch e trasferisci su GPU se necessario
            targets, weights, descr, infgt= batch  # Modifica in base alla tua struttura del batch
            weights, descr, infgt = weights.to(self.device), descr.to(self.device),infgt.to(self.device), targets.to(self.device)
            for i in self.forgetting_subset:
                orig_target = targets[i]
                while targets[i] == orig_target:
                    targets[i] = random.randint(0, 9)
            # Recupera pesi della classe target 
            weights = self.last_layer_weights(targets)
            descr = self.description[targets]

            # Forward pass
            att_from_att, att_from_weight, weight_from_weight, weight_from_att, latent_att, latent_weight = self.joint_ae((descr, weights))

            # Calcola la perdita
            loss = self.compute_loss(descr, att_from_att, weights, weight_from_weight, att_from_weight, weight_from_att)

            # Backward pass
            self.descr_optimizer.zero_grad()
            self.weights_optimizer.zero_grad()
            loss.backward()
            self.descr_optimizer.step()
            self.weights_optimizer.step()
            running_loss += loss.item()
            #LE TRE RIGHE SUCCESSIVE NON SO SE VANNO MESSE O MENO
            self.optimizer.zero_grad()  # Azzerare i gradienti
            loss.backward()  # Calcola i gradienti
            self.optimizer.step()  # Aggiorna i pesi del modello

        print(f"Loss medio di epoca: {running_loss / len(loader)}")

    def compute_loss(self, descr, att_from_att, weights, weight_from_weight, att_from_weight, weight_from_att):
        loss = nn.MSELoss()(descr, att_from_att) + nn.MSELoss()(weights, weight_from_weight) + nn.MSELoss()(weights, weight_from_att) + nn.MSELoss()(descr, att_from_weight)
        return loss

#ACTUALLY NOT USED
    def forward(self, x, target, random_target=None, infgt=0):
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
