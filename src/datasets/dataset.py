import torch
import torchvision
import numpy as np
from torchvision import transforms
import os


class SaliencyDataset(torch.utils.data.Dataset):
    def __init__(self, orig_dataset, transform=None):
        self.orig_dataset = orig_dataset
        self.transform = transform

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

    return train, val, test





if __name__ == "__main__":

    data = [
        'cifar10',
        'cifar100',
        'caltech101',
        #'imagenet',
        'oxford-iiit-pet',
        'oxford-flowers',
        'svhn',
        'mnist',
        'fashionmnist',
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
            plt.imshow(img.permute(1, 2, 0))
            plt.title(lbl)
            plt.show()
    
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

        # print number of classes
        print(f'Number of classes: {len(class_distribution)}')

        # compute the class unbalance as the ratio between the number of samples in the most frequent class and the number of samples in the least frequent class
        dist = list(class_distribution.values())
        class_unbalance = max(dist) / min(dist)
        print(f'Class unbalance: {class_unbalance}')
    '''