import torch
from torch.utils.data import Dataset
import random
#from src.utils import retrieve_weights
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.descr_and_similarity import calculate_embeddings
import requests
from transformers import BertModel, BertTokenizer

class UnlearningDataset(Dataset):
    def __init__(self, dataset, forget_indices):
        """
        UnlearningDataset class.
        Args:
            dataset (Dataset): original image dataset.
            forget_indices (list): index of the samples to forget.
        """
        self.dataset = dataset
        self.forget_indices = set(forget_indices) 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, label = self.dataset[index]
        infgt = 1 if index in self.forget_indices else 0
        return input, label, infgt


class IcusUnlearningDataset(Dataset):
    def __init__(self, orig_dataset, nlayers, infgt, model, num_classes, device="cpu"):
        """
        IcusUnlearningDataset class.
        Args:
            orig_dataset (str): name of the original dataset.
            infgt (Tensor): flag infgt (1 or 0).
            model (nn.Module): model to use.
            num_classes (int): number of classes.
            device (str): device to use.
        """
        self.classes = torch.arange(0, num_classes)
        self.descr = calculate_embeddings(orig_dataset)  # Tensor delle descrizioni
        self.distinct, self.shared = model.get_weights(num_classes, nlayers)  # Tensor dei pesi distinti e condivisi
        self.infgt = infgt  # Tensor dei flag infgt (1 o 0)
        self.device = device

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        # Estrai la classe, i pesi, la descrizione e il flag infgt in base all'indice
        classe = self.classes[idx].to(self.device)
        if len(self.distinct) == 0:
            weigths = self.shared.to(self.device)
        else:
            weigths = torch.cat((self.distinct[idx], self.shared), 0).to(self.device)
        descr = self.descr[idx].to(self.device)
        infgt = self.infgt[idx].to(self.device)
        return classe, weigths, descr, infgt

    


def get_unlearning_dataset(cfg, unlearning_method_name, model, train, retain_indices, forget_indices, forgetting_subset): 
    if unlearning_method_name == 'icus' or unlearning_method_name == 'icus_hierarchy':
        num_classes = cfg.dataset.classes
        infgt = torch.tensor([1 if i in forgetting_subset else 0 for i in range(len(train))])  
        unlearning_train = IcusUnlearningDataset(cfg.dataset.name, cfg.unlearn.nlayers, infgt, model, num_classes, cfg.device)
        unlearning_train = torch.utils.data.DataLoader(unlearning_train, batch_size=cfg.dataset.classes, num_workers=0)
    else:
        unlearning_train = UnlearningDataset(train, forget_indices)
        unlearning_train = torch.utils.data.DataLoader(unlearning_train, batch_size=cfg.train.batch_size, num_workers=8)
    return unlearning_train


