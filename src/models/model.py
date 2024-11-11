import torch
import torchvision
import numpy as np
import os
from src.models.resnet import ResNet9, ResidualBlock


# goodbye and thank you for all the fish: https://huggingface.co/anonauthors 
def load_model(model_name, checkpoint=None, device='cpu'):
    """Load model from checkpoint
    model_name: str: model name
    dataset: str: dataset name
    checkpoint: str: path to checkpoint
    device: str: device to load model
    """
    model = None
    if model_name == 'resnet':
        model=ResNet9(ResidualBlock)
    else:
        raise ValueError(f'Unknown model: {model_name}')
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    return model


if __name__ == '__main__':

    model_list = ['resnet', 'vit']
    dataset_list = ['cifar10', 'cifar100', 'caltech101', 'oxford-pet', 'oxford-flower']
    img = torch.randn(2, 3, 224, 224) # fake batch of RGB images

    for model_name in model_list:
        for dataset in dataset_list:
            print(f'\n\nModel: {model_name} - Dataset: {dataset}')
            model = load_model(model_name, dataset)
            out = model(img)
            print(out.shape)
