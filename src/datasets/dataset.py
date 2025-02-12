import torch
import torchvision
from sklearn.datasets import fetch_lfw_people
import numpy as np
from torchvision import transforms
import os
import sys
import math
from collections import defaultdict
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from scripts.parse_agedb_dataset import retrieve_AgeDB_dataset


class ImgTextDataset(torch.utils.data.Dataset):
    def __init__(self, orig_dataset, transform=None):
        self.orig_dataset = orig_dataset
        self.transform = transform
        self.targets = [lbl for _, lbl in orig_dataset]

    def __len__(self):
        return self.orig_dataset.__len__()

    def __getitem__(self, idx):
        img, lbl = self.orig_dataset.__getitem__(idx)
        img = torch.clamp(img, 0, 1) # clamp the image to be between 0 and 1
        # clip image to be three channels
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        if self.transform:
            img = self.transform(img)

        return img, lbl


def load_dataset(dataset, data_dir, resize=224, val_split=0.2, test_split=0.2):

    train, val, test = None, None, None

    torch.manual_seed(42)
    np.random.seed(42)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((resize, resize)),
    ])

    # CIFAR-10
    if dataset == 'cifar10':
        train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

        split = int(len(train) * val_split)
        train, val = torch.utils.data.random_split(train, [len(train) - split, split])
        test = torch.utils.data.Subset(test, list(range(int(len(test)))))

    # CIFAR-100
    elif dataset == 'cifar100':
        train = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)

        split = int(len(train) * val_split)
        train, val = torch.utils.data.random_split(train, [len(train) - split, split])

    # Caltech101
    elif dataset == 'caltech101':
        data = torchvision.datasets.Caltech101(data_dir, download=True, transform=transform)
        num_train = len(data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        val_split = int(val_split * num_train)
        test_split = int(test_split * num_train)
        train_idx, val_idx, test_idx = indices[val_split+test_split:], indices[:val_split], indices[val_split:val_split+test_split]
        train = torch.utils.data.Subset(data, train_idx)
        val = torch.utils.data.Subset(data, val_idx)
        test = torch.utils.data.Subset(data, test_idx)


    # ImageNet
    elif dataset == 'imagenet':
        imsize = 299

        preprocess = transforms.Compose([
            transforms.Resize((imsize, imsize)),  
            transforms.ToTensor(),  # torch.Tensor 
        ])

        data_dir = '../../data/ILSVRC2012_img_val_subset'
        val = torchvision.datasets.ImageFolder(os.path.join(data_dir), preprocess)
        train = val
        test = val

    # Oxford-IIIT Pet
    elif dataset == 'oxford-iiit-pet':
        data = torchvision.datasets.OxfordIIITPet(data_dir, download=True, transform=transform)
        num_train = len(data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        val_split = int(val_split * num_train)
        test_split = int(test_split * num_train)
        train_idx, val_idx, test_idx = indices[val_split+test_split:], indices[:val_split], indices[val_split:val_split+test_split]
        train = torch.utils.data.Subset(data, train_idx)
        val = torch.utils.data.Subset(data, val_idx)
        test = torch.utils.data.Subset(data, test_idx)

    # Oxford Flowers
    elif dataset == 'oxford-flowers':
        data = torchvision.datasets.Flowers102(data_dir, split='train', download=True, transform=transform)
        val_data = torchvision.datasets.Flowers102(data_dir, split='val', download=True, transform=transform)
        test_data = torchvision.datasets.Flowers102(data_dir, split='test', download=True, transform=transform)
        num_train = len(data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        train = data
        val = val_data
        test = test_data
    
    # LFW dataset
    elif dataset == 'lfw':
        data = fetch_lfw_people(
            color=True,
            resize=resize / 256,  # Imposta la risoluzione
            min_faces_per_person=20  # Prendiamo tutte le classi senza filtro
        )

        images = torch.tensor(data.images).permute(0, 3, 1, 2).float()  # Converti le immagini in tensori
        labels = torch.tensor(data.target)

        print(f"[DEBUG] Numero totale di immagini: {len(images)}")
        print(f"[DEBUG] Numero totale di classi: {len(np.unique(labels.numpy()))}")

        # Raggruppa le immagini per classe
        class_dict = defaultdict(list)
        for img, lbl in zip(images, labels):
            class_dict[lbl.item()].append(img)

        # Inizializza liste per i set di train, validation e test
        train_images, val_images, test_images = [], [], []
        train_labels, val_labels, test_labels = [], [], []

        # Distribuisci le immagini tra train, val e test, assicurando almeno un esempio per classe
        for lbl, imgs in class_dict.items():
            np.random.shuffle(imgs)  # Mescola le immagini per ogni classe

            # Assegna almeno una immagine a train, val e test
            train_images.append(imgs[0])
            train_images.append(imgs[1])
            train_images.append(imgs[2])
            train_images.append(imgs[3])
            val_images.append(imgs[4])
            test_images.append(imgs[5])

            train_labels.append(lbl)
            train_labels.append(lbl)
            train_labels.append(lbl)
            train_labels.append(lbl)
            val_labels.append(lbl)
            test_labels.append(lbl)

            # Distribuisci le immagini rimanenti
            remaining = imgs[6:]
            for i, img in enumerate(remaining):
                if len(train_images) <= len(val_images) and len(train_images) <= len(test_images):
                    train_images.append(img)
                    train_labels.append(lbl)
                elif len(val_images) <= len(test_images):
                    val_images.append(img)
                    val_labels.append(lbl)
                else:
                    test_images.append(img)
                    test_labels.append(lbl)

        # Converti le immagini in tensori
        train_images = torch.stack(train_images)
        val_images = torch.stack(val_images)
        test_images = torch.stack(test_images)

        # Converti le etichette in tensori
        train_labels = torch.tensor(train_labels)
        val_labels = torch.tensor(val_labels)
        test_labels = torch.tensor(test_labels)

        # Debugging: stampa le dimensioni dei set
        print(f"[DEBUG] Numero immagini Train: {len(train_images)}")
        print(f"[DEBUG] Numero immagini Val: {len(val_images)}")
        print(f"[DEBUG] Numero immagini Test: {len(test_images)}")

        # Verifica che ogni classe sia presente in tutti i set
        unique_train = torch.unique(train_labels)
        unique_val = torch.unique(val_labels)
        unique_test = torch.unique(test_labels)

        print(f"[DEBUG] Numero di classi in Train: {len(unique_train)}")
        print(f"[DEBUG] Numero di classi in Val: {len(unique_val)}")
        print(f"[DEBUG] Numero di classi in Test: {len(unique_test)}")

        # Crea un dataset personalizzato per LFW
        class LfwDataset(torch.utils.data.Dataset):
            def __init__(self, images, labels):
                self.images = images
                self.labels = labels

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                return self.images[idx], self.labels[idx]

        train = LfwDataset(train_images, train_labels)
        val = LfwDataset(val_images, val_labels)
        test = LfwDataset(test_images, test_labels)

        print("[INFO] Dataset LFW caricato con successo!\n")

    elif dataset == 'ageDB':
        class AgeDBDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, transform=None):
                self.dataset = dataset
                self.transform = transform

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                img, lbl = self.dataset[idx]
                if self.transform:
                    img = self.transform(img)
                return img, lbl
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
        os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
        dataset = retrieve_AgeDB_dataset("data/AgeDB.zip", "data/AgeDB/AgeDB")
        
        # Indici dei campioni
        num_samples = len(dataset)
        indices = list(range(num_samples))
        np.random.shuffle(indices)

        # Calcola le dimensioni per validation e test set
        val_split = int(val_split * num_samples)
        test_split = int(test_split * num_samples)

        # Dividi i dati in train, val e test
        train_indices = indices[val_split + test_split:]
        val_indices = indices[:val_split]
        test_indices = indices[val_split:val_split + test_split]

        # Crea i dataset separati per train, val e test
        train = torch.utils.data.Subset(dataset, train_indices)
        val = torch.utils.data.Subset(dataset, val_indices)
        test = torch.utils.data.Subset(dataset, test_indices)

        # Applica la trasformazione (se presente)
        if transform:
            train = AgeDBDataset(train, transform)
            val = AgeDBDataset(val, transform)
            test = AgeDBDataset(test, transform)

    # CUB200 -> I don't know if this is the correct way to load the dataset
    elif dataset == 'cub':

        class CUB200Dataset(torch.utils.data.Dataset):
            def __init__(self, dataset, transform=None):
                self.dataset = dataset
                self.transform = transform

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                img, lbl = self.dataset[idx]['image'], self.dataset[idx]['label']
                if self.transform:
                    img = self.transform(img)
                return img, lbl

        train = datasets.load_dataset('Multimodal-Fatima/CUB_train')['train']
        test = datasets.load_dataset('Multimodal-Fatima/CUB_test')['test']

        train = CUB200Dataset(train, transform)
        test = CUB200Dataset(test, transform)
        
        num_train = len(train)  # generate validation set from training set
        indices = list(range(num_train))
        np.random.shuffle(indices)
        val_split = int(val_split * num_train)
        train_idx, val_idx = indices[val_split:], indices[:val_split]
        train = torch.utils.data.Subset(train, train_idx)
        val = torch.utils.data.Subset(train, val_idx)
        
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    train = ImgTextDataset(train)
    val = ImgTextDataset(val)
    test = ImgTextDataset(test)

    return train, val, test


def get_retain_forget_dataloaders(cfg, retain_dataset, forget_dataset):
    if cfg.unlearning_method == 'scrub' or cfg.unlearning_method == 'ssd':
        # need to balance number of steps, so need to have different batch sizes
        retain_batch_size = math.ceil(cfg.train.batch_size * (cfg.dataset.classes - cfg.forgetting_set_size) / cfg.forgetting_set_size)
        forget_batch_size = cfg.train.batch_size
        retain_loader = DataLoader(retain_dataset, batch_size=retain_batch_size, num_workers=8) 
        forget_loader = DataLoader(forget_dataset, batch_size=forget_batch_size, num_workers=8)
    else:
        retain_loader = DataLoader(retain_dataset, batch_size=cfg.train.batch_size, num_workers=8)
        forget_loader = DataLoader(forget_dataset, batch_size=cfg.train.batch_size, num_workers=8)
    return retain_loader, forget_loader



if __name__ == "__main__":

    data = [
        'cifar10',
        'cifar100',
        'caltech101',
        #'imagenet',
        'oxford-iiit-pet',
        'oxford-flowers',
    ]
    
    for dataset in data:
        print(f'\n\nDataset: {dataset}')
        data = load_dataset(dataset, './data')
        print(data[0].__len__(), data[1].__len__(), data[2].__len__())

        test = data[2]
        print(test)
        import matplotlib.pyplot as plt
        for i in range(10):
            img, lbl = test.__getitem__(i)
            print(img.shape, lbl)
    
    # block to compute class distribution and class unbalance
    '''
    for d in data:
        print(f'\nData: {d}')

        for i in range(len(d)):
            _, label = d[i]
            if label not in class_distribution:
                class_distribution[label] = 0
            class_distribution[label] += 1
        
        # sort and print the class distribution
        class_distribution = dict(sorted(class_distribution.items(), key=lambda x: x[1], reverse=True))
        # for key, value in class_distribution.items():
        #     print(f'{key}: {value}')

        print(f'Number of classes: {len(class_distribution)}') # print number of classes

        # compute the class unbalance as the ratio between the number of samples in the most frequent class and the number of samples in the least frequent class
        dist = list(class_distribution.values())
        class_unbalance = max(dist) / min(dist)
        print(f'Class unbalance: {class_unbalance}')
    '''