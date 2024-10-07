import torch
from torch.utils.data import Dataset

class DatasetWrapper(Dataset):
    def __init__(self, dataset, forget_indices):
        """
        DatasetWrapper avvolge un dataset esistente e aggiunge il flag infgt per indicare se un campione deve essere dimenticato.
        Args:
            dataset (torch.utils.data.Dataset): Dataset originale 
            forget_indices (set or list): Indici dei campioni da dimenticare. Per questi indici, `infgt` sarà 1.
        """
        self.dataset = dataset
        self.forget_indices = set(forget_indices) 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Restituisce features, etichetta e infgt per ogni campione.
        Args:
            index (int): Indice del campione.
        Returns:
            tuple: (features, etichetta, infgt), dove infgt è 1 se l'indice è nei forget_indices, altrimenti 0.
        """
        input, label = self.dataset[index]
        infgt = 1 if index in self.forget_indices else 0
        return input, label, infgt

    