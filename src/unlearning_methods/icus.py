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
        print("Description shape: ", self.description.shape)
        flatten_description = self.description.view(nclass, -1)
        print("Flatten description shape: ", flatten_description.shape)
        self.forgetting_subset = forgetting_subset
        weights, bias = retrieve_weights(model)
        self.device = opt.device
        # Concatenazione dei pesi e bias, assicurati che il bias abbia forma corretta
        self.weights = torch.cat((weights, bias.view(-1, 1)), dim=1)
        print("Weights shape: ", self.weights.shape)
        # Ottieni il numero di input features dai pesi
        # Definizione degli autoencoder
        descr_ae = Autoencoder(opt, flatten_description.shape[1], embed_dim=512, num_layers=2)  
        weights_ae = Autoencoder(opt, self.weights.shape[1], embed_dim=512, num_layers=2) 
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
        for epoch in range(self.opt.train.max_epochs):
            print("Epoca: ", epoch)
            self.train_one_epoch(train_loader)

    def train_one_epoch(self, loader):
        print("Inizio epoca di training")
        self.joint_ae.train()  # Imposta il Joint Autoencoder in modalità training
        running_loss = 0.0

        for batch in loader:
            # Estrai il batch e trasferisci su GPU se necessario
            targets, weights, descr, infgt= batch  # Modifica in base alla tua struttura del batch
            weights, descr, infgt, targets = weights.to(self.device), descr.to(self.device),infgt.to(self.device), targets.to(self.device)
            print("Forgetting subset: ", self.forgetting_subset)
            for target in targets:
                orig_target = target
                if target in self.forgetting_subset:
                    while target == orig_target:
                        target = random.randint(0, 9)
            descr = descr.view(descr.size(0), -1)  # Output: torch.Size([10, 2304])


            print(f"Descr shape: {descr.shape}")  # Dovrebbe essere [10, 2304]
            print(f"Weights shape: {weights.shape}")  # Dovrebbe essere [10, 129]
            for i in self.forgetting_subset:
                weights[i] = torch.randn_like(weights[i])
            # Forward pass
            att_from_att, att_from_weight, weight_from_weight, weight_from_att, latent_att, latent_weight = self.joint_ae((descr, weights))
            latent_distance = self.icus_distance(latent_att, latent_weight)
            print(f"Latent distance: {latent_distance}")
            # Calcola la perdita
            ############### DA RIMUOVERE ###############
            descr_retained = descr[torch.tensor([i for i in range(10) if i not in self.forgetting_subset])]
            descr_forgotten = descr[torch.tensor([i for i in range(10) if i in self.forgetting_subset])]
            weights_retained = weights[torch.tensor([i for i in range(10) if i not in self.forgetting_subset])]
            weights_forgotten = weights[torch.tensor([i for i in range(10) if i in self.forgetting_subset])]
            att_from_att_ret=att_from_att[torch.tensor([i for i in range(10) if i not in self.forgetting_subset])]
            att_from_att_forg=att_from_att[torch.tensor([i for i in range(10) if i in self.forgetting_subset])]
            weights_from_weights_ret=weight_from_weight[torch.tensor([i for i in range(10) if i not in self.forgetting_subset])]
            weights_from_weights_forg=weight_from_weight[torch.tensor([i for i in range(10) if i in self.forgetting_subset])]
            att_from_weight_ret=att_from_weight[torch.tensor([i for i in range(10) if i not in self.forgetting_subset])]
            att_from_weight_forg=att_from_weight[torch.tensor([i for i in range(10) if i in self.forgetting_subset])]
            weight_from_att_ret=weight_from_att[torch.tensor([i for i in range(10) if i not in self.forgetting_subset])]
            weight_from_att_forg=weight_from_att[torch.tensor([i for i in range(10) if i in self.forgetting_subset])]
            retain_loss=self.compute_loss(descr_retained, att_from_att_ret, weights_retained, weights_from_weights_ret, att_from_weight_ret, weight_from_att_ret)
            forget_loss=self.compute_loss(descr_forgotten, att_from_att_forg, weights_forgotten, weights_from_weights_forg, att_from_weight_forg, weight_from_att_forg)
            print(f"Retain loss: {retain_loss}")
            print(f"Forget loss: {forget_loss}")
            ##########################################
            loss = self.compute_loss(descr, att_from_att, weights, weight_from_weight, att_from_weight, weight_from_att)
            loss += latent_distance.mean()
            # Backward pass
            self.descr_optimizer.zero_grad()
            self.weights_optimizer.zero_grad()
            loss.backward()
            self.descr_optimizer.step()
            self.weights_optimizer.step()
            running_loss += loss.item()
            #LE DUE RIGHE SUCCESSIVE NON SO SE VANNO MESSE O MENO
            self.optimizer.zero_grad()  # Azzerare i gradienti
            self.optimizer.step()  # Aggiorna i pesi del modello

        print(f"Loss medio di epoca: {running_loss / len(loader)}")

    def compute_loss(self, descr, att_from_att, weights, weight_from_weight, att_from_weight, weight_from_att):
        loss = nn.MSELoss()(descr, att_from_att) + nn.MSELoss()(weights, weight_from_weight) + nn.MSELoss()(weights, weight_from_att) + nn.MSELoss()(descr, att_from_weight)
        return loss

#ACTUALLY NOT USED
    """def forward(self, x, target, random_target=None, infgt=0):
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

        return loss  # Restituisci la perdita"""
