import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from torch.utils.data import DataLoader
from src.architectures.autoencoder import Autoencoder
from src.architectures.autoencoder import JointAutoencoder
from src.unlearning_methods.base import BaseUnlearningMethod
from src.utils import retrieve_weights

class Icus(BaseUnlearningMethod):
    def __init__(self, opt, model, input_dim, nclass, wrapped_train_loader, forgetting_subset, val_loader, logger=None):
        super().__init__(opt, model)
        self.fc = model.fc
        self.orig_model = copy.deepcopy(self.model)
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
        self.descr_optimizer = optim.Adam(self.joint_ae.ae_d.parameters(), lr=0.0012)
        self.weights_optimizer = optim.Adam(self.joint_ae.ae_w.parameters(), lr=0.0012) #0.00125 -> 63.1%
        #self.optimizer = optim.Adam(self.model.fc.parameters(), lr=0.001)
        

    def last_layer_weights(self, target):
        return self.fc.weight[target]

    def icus_distance(self, weights_encoded, descr_encoded):
        return nn.functional.cosine_similarity(weights_encoded, descr_encoded)

    
    def unlearn(self, model, train_loader):
        self.model.fc.weight.requires_grad_(True)
        self.current_step = 0
        for epoch in range(self.opt.max_epochs):
            print("Epoca: ", epoch)
            self.train_one_epoch(train_loader, epoch) #anche val_loader
            self.test_unlearning_effect(self.val_loader, self.forgetting_subset, epoch, False)
        return self.model


    def train_one_epoch(self, loader, epoch):
        print("Inizio epoca di training")
        self.joint_ae.train()  # Joint Autoencoder in training mode
        self.model.fc.train()  # Model in training mode

        print(self.model.fc.weight.requires_grad)
        print(self.model.fc.bias.requires_grad) 

        running_loss = 0.0

        for batch in loader:
            targets, weights, descr, infgt = batch
            weights, descr, infgt, targets = weights.to(self.device), descr.to(self.device), infgt.to(self.device), targets.to(self.device)
            
            #If necessary, change the target to a random one
            for target in targets:
                orig_target = target
                if target in self.forgetting_subset:
                    while target == orig_target:
                        target = random.randint(0, 9)

            descr = descr.view(descr.size(0), -1)  # Output: torch.Size([10, 2304])

            # Update the weights of the model
            for i in self.forgetting_subset:
                if self.opt.forgetting_set_strategy == "random_values":
                    weights[i] = torch.randn_like(weights[i], requires_grad=True)
                elif self.opt.forgetting_set_strategy == "random_class":
                    j=i
                    while j in self.forgetting_subset:
                        j = random.randint(0, 9)
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
            self.logger.log_metrics({"method":"Icus", "loss": loss.item()}, step=self.current_step)

            # Backward pass
            self.descr_optimizer.zero_grad()
            self.weights_optimizer.zero_grad()
            #self.optimizer.zero_grad() 

            loss.backward()  # Backpropagation
            self.descr_optimizer.step()  # Update autoencoder
            self.weights_optimizer.step() 
            #self.optimizer.step()  # Update model weights

            running_loss += loss.item()

        print(f"Loss medio di epoca: {running_loss / len(loader)}")
        self.logger.log_metrics({"method":"Icus", "average_loss": running_loss / len(loader)}, step=epoch)
        


    def compute_loss(self, descr, att_from_att, weights, weight_from_weight, att_from_weight, weight_from_att, latent_att, latent_weight):
        loss = nn.MSELoss()(descr, att_from_att) + nn.MSELoss()(weights, weight_from_weight) + \
               nn.MSELoss()(weights, weight_from_att) + nn.MSELoss()(descr, att_from_weight) 
        
        return loss 


    def test_unlearning_effect(self, loader, forgetting_subset, epoch, test = True):
        self.orig_model.eval()  # Test mode for the model
        self.joint_ae.eval()  # Test mode for autoencoder

        correct_retain = 0
        total_retain = 0
        correct_forget = 0
        total_forget = 0
        total = 0
        correct = 0
        improved = 0
        with torch.no_grad():  # Disable backpropagation during test
            for batch in loader:

                if test:
                    images, labels = batch
                    images, labels = images.to(self.device), labels.to(self.device)
                else:
                    images, labels, _ = batch
                    images, labels = images.to(self.device), labels.to(self.device)

                # Predictions
                outputs = self.orig_model(images)
                _, predicted = torch.max(outputs, 1)  
                predicted = predicted.to(self.device)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for i, pred in enumerate(predicted):
                    # Embeddings
                    descr_embedding = self.description[pred]  # Usa la descrizione associata alla classe corrente
                    descr_embedding = descr_embedding.view(1, -1).to(self.device)  # Reshape per essere compatibile

                    # Class weights
                    original_weight = self.orig_model.fc.weight[pred]  # Pesi corrispondenti alla classe
                    original_bias = self.orig_model.fc.bias[pred]  # Bias corrispondente alla classe
                    original_weight = torch.cat((original_weight, original_bias.unsqueeze(0)), dim=0)  # Concatena il bias

                    # ICUS phase
                    _, _, weight_from_weight, weight_from_att, _, _ = self.joint_ae((descr_embedding, original_weight.unsqueeze(0), self.opt.device))

                    # ICUS classification
                    weights_transposed = self.orig_model.fc.weight.t()  #Weights matrix transposed (n_feature, n_classi)
                    bias = self.orig_model.fc.bias  # Bias vector (n_classi,)
                    bias = bias.unsqueeze(0)  

                    weights_with_bias = torch.cat((weights_transposed, bias), dim=0)
                    modified_output = torch.matmul(weight_from_att, weights_with_bias)
                    modified_class = torch.argmax(modified_output, dim=1)

                    # Accuracy calculation
                    if labels[i] in forgetting_subset:
                        total_forget += 1
                        if modified_class == labels[i]:
                            correct_forget += 1
                    else:
                        total_retain += 1
                        if modified_class == labels[i]:
                            correct_retain += 1
                        if labels[i] != pred and modified_class == labels[i]:
                            improved += 1

        # Print results
        accuracy_retain = 100 * correct_retain / total_retain
        accuracy_forget = 100 * correct_forget / total_forget
        self.logger.log_metrics({"method":"Icus", "accuracy_retain": accuracy_retain, "accuracy_forget": accuracy_forget}, step=epoch)
        print("Forgetting subset:", forgetting_subset)
        print("# Retain set:", total_retain, " Correct:", correct_retain)
        print("# Forget set:", total_forget, " Correct:", correct_forget)
        if test:
            print(f'Accuracy retain ICUS on test set: {accuracy_retain:.2f}%')
            print(f'Accuracy forget ICUS on test set: {accuracy_forget:.2f}%')
        else:
            print(f'Accuracy retain ICUS on validation set: {accuracy_retain:.2f}%')
            print(f'Accuracy forget ICUS on validation set: {accuracy_forget:.2f}%')
        print(f'Improved accuracy: {improved}')
        return accuracy_retain, accuracy_forget
