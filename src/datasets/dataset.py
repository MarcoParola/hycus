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


def load_dataset(dataset, data_dir, resize=224, val_split=0.125, test_split=0.125):

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
    
    # LFW dataset
    elif dataset == 'lfw':
        data = fetch_lfw_people(
            color=True,
            resize=resize / 256,  # set resolution
            min_faces_per_person=20  # Take only people with at least 20 images
        )

        images = torch.tensor(data.images).permute(0, 3, 1, 2).float()  # Convert to tensor and permute dimensions
        labels = torch.tensor(data.target)

        class_dict = defaultdict(list)
        for img, lbl in zip(images, labels):
            class_dict[lbl.item()].append(img)

        # Initialize lists for train, val and test images and labels
        train_images, val_images, test_images = [], [], []
        train_labels, val_labels, test_labels = [], [], []

        # Distribute the images in train, val and test
        for lbl, imgs in class_dict.items():
            np.random.shuffle(imgs)  

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

            # Distribute the remaining images in train, val and test
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

        # Convert images to tensor
        train_images = torch.stack(train_images)
        val_images = torch.stack(val_images)
        test_images = torch.stack(test_images)

        # convert labels to tensor
        train_labels = torch.tensor(train_labels)
        val_labels = torch.tensor(val_labels)
        test_labels = torch.tensor(test_labels)

        # verify the number of classes in train, val and test
        unique_train = torch.unique(train_labels)
        unique_val = torch.unique(val_labels)
        unique_test = torch.unique(test_labels)

        # create the dataset
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
        
        # sample the dataset
        num_samples = len(dataset)
        indices = list(range(num_samples))
        np.random.shuffle(indices)

        # calculate the number of samples for each split
        val_split = int(val_split * num_samples)
        test_split = int(test_split * num_samples)

        # Dividi i dati in train, val e test
        train_indices = indices[val_split + test_split:]
        val_indices = indices[:val_split]
        test_indices = indices[val_split:val_split + test_split]

        # create the subsets
        train = torch.utils.data.Subset(dataset, train_indices)
        val = torch.utils.data.Subset(dataset, val_indices)
        test = torch.utils.data.Subset(dataset, test_indices)
        

        # Apply the transform
        if transform:
            train = AgeDBDataset(train, transform)
            val = AgeDBDataset(val, transform)
            test = AgeDBDataset(test, transform)

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
        'lfw',
        'ageDB',
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
    