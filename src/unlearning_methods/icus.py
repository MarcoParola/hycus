import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import copy
from torch.utils.data import DataLoader
from src.architectures.autoencoder import Autoencoder
from src.architectures.autoencoder import JointAutoencoder
from src.unlearning_methods.base import BaseUnlearningMethod
from src.utils import retrieve_weights
from src.metrics.metrics import compute_metrics

class Icus(BaseUnlearningMethod):
    def __init__(self, opt, model, input_dim, nclass, wrapped_train_loader, forgetting_subset, val_loader, logger=None):
        super().__init__(opt, model)
        self.fc = model.fc
        self.orig_model = copy.deepcopy(self.model)
        self.wrapped_train_loader = wrapped_train_loader
        self.logger = logger
        self.val_loader = val_loader
        # Inizialize desriptions and other stuff
        self.description = wrapped_train_loader.dataset.descr
        flatten_description = self.description.view(nclass, -1)
        self.forgetting_subset = forgetting_subset
        weights, bias = retrieve_weights(model)
        self.device = opt.device
        # Concatenation weights and bias
        self.weights = torch.cat((weights, bias.view(-1, 1)), dim=1)
        #autoencoder
        descr_ae = Autoencoder(opt, flatten_description.shape[1], embed_dim=512, num_layers=2)  
        descr_ae.to(opt.device)
        weights_ae = Autoencoder(opt, self.weights.shape[1], embed_dim=512, num_layers=2) 
        weights_ae.to(opt.device)
        # Joint Autoencoder
        self.joint_ae = JointAutoencoder(descr_ae, weights_ae, self.opt.device)
        self.current_step = 0
        # Autoencoder optimizers
        self.descr_optimizer = optim.Adam(self.joint_ae.ae_d.parameters(), lr=1e-3)
        self.weights_optimizer = optim.Adam(self.joint_ae.ae_w.parameters(), lr=1e-3)
        
        

    def last_layer_weights(self, target):
        return self.fc.weight[target]

    def icus_distance(self, weights_encoded, descr_encoded):
        return nn.functional.cosine_similarity(weights_encoded, descr_encoded)

    
    def unlearn(self, model, train_loader):
        self.model.fc.weight.requires_grad_(True)
        self.current_step = 0
        for epoch in range(self.opt.max_epochs):
            print("Epoch: ", epoch)
            self.train_one_epoch(train_loader, epoch) 
            self.test_unlearning_effect(self.wrapped_train_loader, self.val_loader, self.forgetting_subset, epoch, False)
        return self.model


    def train_one_epoch(self, loader, epoch):
        print("Train epoch start")
        self.joint_ae.train()  # Joint Autoencoder in training mode
        self.model.fc.train()  # Model in training mode

        running_loss = 0.0

        for batch in loader:
            targets, weights, descr, infgt = batch
            weights, descr, infgt, targets = weights.to(self.device), descr.to(self.device), infgt.to(self.device), targets.to(self.device)

            descr = descr.view(descr.size(0), -1)  # Output: torch.Size([10, 2304])

            # Update the weights of the model
            for i in self.forgetting_subset:
                if self.opt.forgetting_set_strategy == "random_values":
                    weights[i] = torch.randn_like(weights[i], requires_grad=True)
                elif self.opt.forgetting_set_strategy == "random_class":
                    j = random.choice([x for x in range(10) if x not in self.forgetting_subset])
                    weights[i] = weights[j]
                elif self.opt.forgetting_set_strategy == "zeros":
                    weights[i] = torch.zeros_like(weights[i], requires_grad=True)
                else:
                    raise ValueError("Invalid forgetting set strategy")
            
            weights.requires_grad_(True)
            descr.requires_grad_(True)

            # Forward pass
            att_from_att, att_from_weight, weight_from_weight, weight_from_att, latent_att, latent_weight = self.joint_ae((descr, weights, self.opt.device))

            loss = self.compute_loss(descr, att_from_att, weights, weight_from_weight, att_from_weight, weight_from_att, latent_att, latent_weight)
            self.logger.log_metrics({"loss": loss.item()}, step=self.current_step)

            # Backward pass
            self.descr_optimizer.zero_grad()
            self.weights_optimizer.zero_grad()

            loss.backward()  # Backpropagation
            self.descr_optimizer.step()  # Update autoencoder
            self.weights_optimizer.step() 

            running_loss += loss.item()

        print(f"Mean loss in this epoch: {running_loss / len(loader)}")
        self.logger.log_metrics({"average_loss": running_loss / len(loader)}, step=epoch)
        


    def compute_loss(self, descr, att_from_att, weights, weight_from_weight, att_from_weight, weight_from_att, latent_att, latent_weight):
        loss = nn.MSELoss()(descr, att_from_att) + nn.MSELoss()(weights, weight_from_weight) + \
               nn.MSELoss()(weights, weight_from_att) + nn.MSELoss()(descr, att_from_weight) #+ 1 - torch.mean(F.cosine_similarity(latent_att, latent_weight))
        print("Cosine similarity: ", 1 - torch.mean(F.cosine_similarity(latent_att, latent_weight)))
        return loss 


    def test_unlearning_effect(self, wrapped_loader, loader, forgetting_subset, epoch, test = True):
        self.model.eval()
        self.joint_ae.eval()
        for i in range(self.opt.dataset.classes):
            _,w,d,_ = wrapped_loader.dataset[i]
            d = d.view(-1)
            latent_w = self.joint_ae.ae_w.encode(w.to(self.opt.device))
            latent_d = self.joint_ae.ae_d.encode(d.to(self.opt.device))
            w = self.joint_ae.ae_w.decode(latent_w)
            weights = w[:-1]
            bias = w[-1]
            with torch.no_grad():
                self.model.fc.weight[i] = weights
                self.model.fc.bias[i] = bias
        metrics = compute_metrics(self.model,loader,self.opt.dataset.classes,forgetting_subset, test) 
        self.logger.log_metrics({'accuracy_retain': metrics['accuracy_retaining'], 'accuracy_forget': metrics['accuracy_forgetting']})
        print("Accuracy forget ", metrics['accuracy_forgetting'])
        print("Accuracy retain ", metrics['accuracy_retaining'])
        