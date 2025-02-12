import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import random
import copy
import json
from torch.utils.data import DataLoader
from src.unlearning_methods.base import BaseUnlearningMethod
from src.utils import retrieve_weights, get_numbers_from_superclass
from src.metrics.metrics import compute_metrics

class Autoencoder(nn.Module):
    def __init__(self, opt, input_dim, embed_dim, output_dim=None, num_layers=3, vae=False, bias=True):
        super(Autoencoder, self).__init__()
        self.opt = opt
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim   
        self.embed_dim = [2 * embed_dim, embed_dim] if vae else [embed_dim, embed_dim]

        if num_layers == 2:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.embed_dim[0]),
                nn.ReLU(inplace=True) if not vae else nn.Identity(inplace=True)
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.embed_dim[0], self.output_dim)
            )
        elif num_layers == 3:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.embed_dim[0]),
                nn.ReLU(inplace=True) if not vae else nn.Identity(inplace=True)
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.embed_dim[1], 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000, self.output_dim)
            )
        elif num_layers == 4:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.embed_dim[0]),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim[0], self.embed_dim[0]),
                nn.ReLU(inplace=True) if not vae else nn.Identity(inplace=True)
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.embed_dim[1], 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000, self.output_dim)
            )

    def encode(self, x):
        return self.encoder(x)  
    def decode(self, x):
        return self.decoder(x)  

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z)
    
class JointAutoencoder(nn.Module):
    def __init__(self, autoencoder_descr, autoencoder_weight, device):
        super(JointAutoencoder, self).__init__()
        self.ae_d = autoencoder_descr
        self.ae_w = autoencoder_weight
        self.device = device
    
    def encode_descr(self, x): 
        return self.ae_d.encode(x)

    def encode_weight(self, x):
        return self.ae_w.encode(x)

    def decode_descr(self, x):
        return self.ae_d.decode(x)

    def decode_weight(self, x):
        return self.ae_w.decode(x)
        
    def forward(self, x):
        descr_in, weight_in, _ = x  
        descr_in = descr_in.to(self.device)  
        weight_in = weight_in.to(self.device)
        latent_descr = self.encode_descr(descr_in)
        latent_weight = self.encode_weight(weight_in)
        
        descr_from_descr = self.decode_descr(latent_descr)
        descr_from_weight = self.decode_descr(latent_weight)
        weight_from_weight = self.decode_weight(latent_weight)
        weight_from_descr = self.decode_weight(latent_descr)
        
        return descr_from_descr, descr_from_weight, weight_from_weight, weight_from_descr, latent_descr, latent_weight

def aggregate_shared(shared_parts, method):
    stacked = torch.stack(shared_parts) 
    if method == "mean":
        return torch.mean(stacked, dim=0)
    elif method == "min":
        return torch.min(stacked, dim=0).values
    elif method == "max":
        return torch.max(stacked, dim=0).values
    elif method == "median":
        return torch.median(stacked, dim=0).values
    else:
        raise ValueError(f"Aggregation meyhod '{method}' not supported.")

class Icus(BaseUnlearningMethod):
    def __init__(self, opt, model, input_dim, nclass, wrapped_train_loader, forgetting_subset, logger):
        super().__init__(opt, model)
        self.opt=opt
        self.orig_model = copy.deepcopy(self.model)
        self.wrapped_train_loader = wrapped_train_loader
        self.logger = logger
        self.description = wrapped_train_loader.dataset.descr
        flatten_description = self.description.view(nclass, -1)
        self.forgetting_subset = forgetting_subset
        self.weights = []
        self.orig_weights = []
        for classe, weights, _, _ in wrapped_train_loader:
            for i in range(len(classe)):
                self.weights.append(weights[i])
        
        self.orig_weights = copy.deepcopy(self.weights)
        self.device = opt.device
        #autoencoder
        descr_ae = Autoencoder(opt, flatten_description.shape[1], embed_dim=512, num_layers=2)  
        descr_ae.to(opt.device)
        input_dim = weights.size(1)
        weights_ae = Autoencoder(opt, input_dim, embed_dim=512, num_layers=2) 
        weights_ae.to(opt.device)
        # Joint Autoencoder
        self.joint_ae = JointAutoencoder(descr_ae, weights_ae, self.opt.device)
        self.current_step = 0
        # Autoencoder optimizers
        self.descr_optimizer = optim.Adam(self.joint_ae.ae_d.parameters(), lr=opt.unlearn.lr)
        self.weights_optimizer = optim.Adam(self.joint_ae.ae_w.parameters(), lr=opt.unlearn.lr)
        
        
    def last_layer_weights(self, target):
        return self.fc.weight[target]

    def icus_distance(self, weights_encoded, descr_encoded):
        return nn.functional.cosine_similarity(weights_encoded, descr_encoded)

    
    def unlearn(self, model, unlearning_train, val_loader):
        self.current_step = 0
        for epoch in range(self.opt.unlearn.max_epochs):
            print("Epoch: ", epoch)
            self.train_one_epoch(unlearning_train, val_loader, epoch) 
        return self.model


    def train_one_epoch(self, unlearning_train, val_loader, epoch):
        print("Train epoch start")
        self.joint_ae.train()  # Joint Autoencoder in training mode
        self.model.model.fc.train()  # Model in training mode
        current_batch=0
        running_loss = 0.0
        for batch in unlearning_train:
            torch.cuda.empty_cache()
            targets, weights, descr, _ = batch
            weights, descr, targets = weights.to(self.device), descr.to(self.device), targets.to(self.device)
            
            descr = descr.view(descr.size(0), -1)
            # Update the weights of the model
            for i in self.forgetting_subset:
                if i >= (current_batch+1) * len(targets) or i < current_batch * len(targets):
                    continue
                i = i % len(targets)
                q = i // len(targets)
                if self.opt.forgetting_set_strategy == "random_values":
                    if targets == q * len(targets) + i:
                        weights[i] = torch.randn_like(weights[i], requires_grad=True)
                elif self.opt.forgetting_set_strategy == "random_class":
                    if 1 not in self.opt.unlearn.nlayers:
                        weights[i] = torch.randn_like(weights[i], requires_grad=True)
                    else:
                        j = random.choice([x for x in range(self.opt.dataset.classes) if x not in self.forgetting_subset])
                        weights[i][:self.model.model.fc[0].weight.data[i].size(0)] = self.orig_weights[j][:self.model.model.fc[0].weight.data[i].size(0)]
                        weights[i][self.model.model.fc[0].weight.data[i].size(0):] = torch.randn_like(weights[i][self.model.model.fc[0].weight.data[i].size(0):])
                elif self.opt.forgetting_set_strategy == "zeros":
                    if targets == q * self.opt.train.batch_size + i:
                        weights[i][:self.model.model.fc[0].weight.data[i].size(0)] = torch.zeros_like(weights[i][:self.model.model.fc[0].weight.data[i].size(0)], requires_grad=True)
                        weights[i][self.model.model.fc[0].weight.data[i].size(0):] = torch.randn_like(weights[i][self.model.model.fc[0].weight.data[i].size(0):])
                else:
                    raise ValueError("Invalid forgetting set strategy")
            current_batch+=1
            
            weights.requires_grad_(True)
            descr.requires_grad_(True)
            
            # Forward pass
            att_from_att, att_from_weight, weight_from_weight, weight_from_att, latent_att, latent_weight = self.joint_ae((descr, weights, self.opt.device))

            loss = self.compute_loss(descr, att_from_att, weights, weight_from_weight, att_from_weight, weight_from_att, latent_att, latent_weight)
            self.logger.log_metrics({"loss": loss.item()}, step=self.current_step)

            # Backward pass
            self.descr_optimizer.zero_grad()
            self.weights_optimizer.zero_grad()
            loss.backward()
            self.descr_optimizer.step()  # Update autoencoder
            self.weights_optimizer.step() 

            running_loss += loss.item()

        print(f"Mean loss in this epoch: {running_loss / len(unlearning_train)}")
        self.logger.log_metrics({"average_loss": running_loss / len(unlearning_train)}, step=epoch)
        if self.opt.dataset.name=='cifar100':
            interval_log=100
        else:
            interval_log=100

        if epoch % interval_log == 0 or epoch == self.opt.unlearn.max_epochs-1:
            self.test_unlearning_effect(unlearning_train, val_loader, self.forgetting_subset, epoch)
        

    def compute_loss(self, descr, att_from_att, weights, weight_from_weight, att_from_weight, weight_from_att, latent_att, latent_weight):
        loss = nn.MSELoss()(descr, att_from_att) + nn.MSELoss()(weights, weight_from_weight) + \
               nn.MSELoss()(weights, weight_from_att) + nn.MSELoss()(descr, att_from_weight) + \
               self.opt.unlearn.cos_sim_factor * (1 - torch.mean(F.cosine_similarity(latent_att, latent_weight))) 
               #+self.opt.unlearn.latent_reg_factor * nn.MSELoss()(latent_att, latent_weight)
        self.logger.log_metrics({"att_from_att_loss": nn.MSELoss()(descr, att_from_att).item()},step=self.current_step)
        self.logger.log_metrics({"weights_from_weights_loss": nn.MSELoss()(weights, weight_from_weight).item()},step=self.current_step)
        self.logger.log_metrics({"weights_from_att_loss": nn.MSELoss()(weights, weight_from_att).item()},step=self.current_step)
        self.logger.log_metrics({"att_from_weights_loss": nn.MSELoss()(descr, att_from_weight).item()},step=self.current_step)
        self.logger.log_metrics({"cosine_similarity loss": 1 - torch.mean(F.cosine_similarity(latent_att, latent_weight)).item()},step=self.current_step)
        return loss 


    def test_unlearning_effect(self, wrapped_loader, loader, forgetting_subset, epoch):
        self.model.eval()
        self.joint_ae.eval()
        distinct = []
        shared_parts = [] 
        
        for i in range(self.opt.dataset.classes):
            _, w, d, _ = wrapped_loader.dataset[i]
            d = d.view(-1)
            latent_w = self.joint_ae.ae_w.encode(w.to(self.opt.device))
            latent_d = self.joint_ae.ae_d.encode(d.to(self.opt.device))
            
            if self.opt.unlearn.reconstruct_from_d:
                w = self.joint_ae.ae_w.decode(latent_d)
            else:
                w = self.joint_ae.ae_w.decode(latent_w)
            
            if 1 in self.opt.unlearn.nlayers: 
                distinct.append(w[:self.model.model.fc[0].weight.size(1) + 1])
                shared_parts.append(w[self.model.model.fc[0].weight.size(1) + 1:])
            else:
                shared_parts.append(w)
        
        aggregation_method = self.opt.unlearn.aggregation_method
        fs = self.opt.forgetting_set 
        os.makedirs(f"shared_weights/forgetting_set_{fs}/{aggregation_method}", exist_ok=True)
        torch.save(shared_parts, f"shared_weights/forgetting_set_{fs}/{aggregation_method}/shared_weights_epoch_{epoch}.pt")
        shared = aggregate_shared(shared_parts, aggregation_method).to(self.opt.device)
        
        if distinct:
            distinct = torch.stack(distinct).to(self.opt.device)
        
        nlayers = self.opt.unlearn.nlayers
        self.model.set_weights(distinct, shared, self.opt.dataset.classes, nlayers)
        metrics = compute_metrics(self.model, loader, self.opt.dataset.classes, forgetting_subset)
        
        # Log delle metriche
        self.logger.log_metrics({'accuracy_retain': metrics['accuracy_retaining'], 'accuracy_forget': metrics['accuracy_forgetting']})


class IcusHierarchy(Icus):
    def __init__(self, semantic_dict, opt, model, input_dim, nclass, wrapped_train_loader, forgetting_subset, logger):
        super().__init__(opt, model, input_dim, nclass, wrapped_train_loader, forgetting_subset, logger)
        self.semantic_dict = semantic_dict
        self.logger = logger
    

    def train_one_epoch(self, unlearning_train, val_loader, epoch):
        print("Train epoch start")
        self.joint_ae.train()  # Joint Autoencoder in training mode
        self.model.model.fc.train()  # Model in training mode

        running_loss = 0.0
        for batch in unlearning_train:
            targets, weights, descr, _ = batch
            weights, descr, targets = weights.to(self.device), descr.to(self.device), targets.to(self.device)

            descr = descr.view(descr.size(0), -1)
            # Update the weights of the model
            for i in self.forgetting_subset:
                if self.opt.forgetting_set_strategy == "random_values":
                    if targets == i:
                        weights[i] = torch.randn_like(weights[i], requires_grad=True)
                elif self.opt.forgetting_set_strategy == "random_class":
                    if 1 not in self.opt.unlearn.nlayers:
                        weights[i] = torch.randn_like(weights[i], requires_grad=True)
                    else:
                        list_of_superclass = get_numbers_from_superclass(i, self.semantic_dict)
                        if list_of_superclass == []:
                            j = random.choice([x for x in range(self.opt.dataset.classes) if x not in self.forgetting_subset])
                        else:
                            j = random.choice(list_of_superclass)
                        weights[i][:self.model.model.fc[0].weight.data[i].size(0) + 1] = weights[j][:self.model.model.fc[0].weight.data[i].size(0) + 1]
                        torch.cat((weights[i], torch.randn_like(weights[i][self.model.model.fc[0].weight.data[i].size(0) + 1:])), dim=0)
                elif self.opt.forgetting_set_strategy == "zeros":
                    if targets == i:
                        weights[i][:self.model.model.fc[0].weight.data[i].size(0) + 1] = torch.zeros_like(weights[i][:self.model.model.fc[0].weight.data[i].size(0) + 1], requires_grad=True)
                        torch.cat((weights[i], torch.randn_like(weights[i][self.model.model.fc[0].weight.data[i].size(0) + 1:])), dim=0)
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

            loss.backward()
            self.descr_optimizer.step()  # Update autoencoder
            self.weights_optimizer.step() 

            running_loss += loss.item()

        print(f"Mean loss in this epoch: {running_loss / len(unlearning_train)}")
        self.logger.log_metrics({"average_loss": running_loss / len(unlearning_train)}, step=epoch)
        if epoch%100 == 0:
            self.test_unlearning_effect(unlearning_train, val_loader, self.forgetting_subset, epoch)
