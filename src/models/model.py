import torch
import torchvision
import numpy as np
import os
from src.models.resnet import ResNet9, ResNet18, ResNetCustom, ResidualBlock
from src.models.classifier import Classifier

# goodbye and thank you for all the fish: https://huggingface.co/anonauthors 
def load_model(weights_name, checkpoint=None, device='cpu'):
    model = Classifier(weights_name, 10, finetune=True)
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
