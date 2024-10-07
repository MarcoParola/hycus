from torch.utils.data import Dataset

class PesiDescrClasseInfgtDataset(Dataset):
    def __init__(self, classi, pesi, descrizioni, infgt, cuda=False, orig_dataset="cifar10"):
        """
        Args:
            classi (Tensor): Tensor contenente le etichette delle classi.
            pesi (Tensor): Tensor contenente i pesi associati a ciascuna classe.
            descrizioni (Tensor): Tensor contenente le descrizioni (o embedding) delle classi.
            infgt_flags (Tensor): Tensor contenente i flag infgt per ciascuna classe (0 o 1).
            cuda (bool): Se True, carica i dati su GPU.
        """
        assert len(classi) == len(pesi) == len(descrizioni) == len(infgt), "Le lunghezze non corrispondono!"
        self.classi = classi  # Tensor delle classi
        self.pesi = pesi  # Tensor dei pesi
        self.descrizioni = descrizioni  # Tensor delle descrizioni
        self.infgt = infgt  # Tensor dei flag infgt (1 o 0)
        self.cuda = cuda
        self.orig_dataset = orig_dataset

    def __len__(self):
        return len(self.classi)

    def __getitem__(self, idx):
        # Estrai la classe, i pesi, la descrizione e il flag infgt in base all'indice
        classe = self.classi[idx]
        pesi = self.pesi[idx]
        descrizione = self.descrizioni[idx]
        infgt = self.infgt[idx]

        # Se richiesto, trasferisci su GPU
        if self.cuda:
            classe = classe.cuda()
            pesi = pesi.cuda()
            descrizione = descrizione.cuda()
            infgt = infgt.cuda()

        # Restituisce la tupla (classe, pesi, descrizione, infgt)
        return classe, pesi, descrizione, infgt


    def find_embedding(self, classe):
        """
        Restituisce l'embedding associato a una classe.
        Args:
            classe (int): Classe di cui trovare l'embedding.
        Returns:
            Tensor: L'embedding associato alla classe.
        """
        if self.orig_dataset == "cifar10":
            filename = "../data/cifar10_classes.txt"
        else:
            print("Dataset non supportato")
        with open(filename, 'r') as f:
            classi = [line.strip() for line in f.readlines()]
        

            

