import torch
import torchvision
import numpy as np
import os
import timm


# goodbye and thank you for all the fish: https://huggingface.co/anonauthors 
def load_model(model_name, dataset): # TODO add checkpoints as optional parameter (..., checkpoint=None), and add device as optional parameter (..., device='cpu')
    """Load a model from the Hugging Face model hub
    model_name: str: model name
    dataset: str: dataset name
    """
    model = None

    # CIFAR-10
    if dataset == 'cifar10':
        if model_name == 'resnet':
            model = timm.create_model("hf-hub:anonauthors/cifar10-timm-resnet50", pretrained=True)
        elif model_name == 'vit':
            model = timm.create_model("hf-hub:anonauthors/cifar10-timm-vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=True)

    # CIFAR-100
    elif dataset == 'cifar100':
        if model_name == 'resnet':
            model = timm.create_model("hf-hub:anonauthors/cifar100-timm-resnet50", pretrained=True)
        elif model_name == 'vit':
            model = timm.create_model("hf-hub:anonauthors/cifar100-timm-vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=True)
    
    # Caltech101
    elif dataset == 'caltech101':
        if model_name == 'resnet':
            model = timm.create_model('hf-hub:anonauthors/caltech101-timm-resnet50', pretrained=True)
        elif model_name == 'vit':
            model = timm.create_model("hf-hub:anonauthors/caltech101-timm-vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=True)

    # ImageNet
    elif dataset == 'imagenet':
        if model_name == 'resnet':
            pass # TODO 
        elif model_name == 'vit':
            pass # TODO

    # Oxford-IIIT Pet
    elif dataset == 'oxford-iiit-pet':
        if model_name == 'resnet':
            model = timm.create_model('hf-hub:anonauthors/oxford_pet-timm-resnet50', pretrained=True)
        elif model_name == 'vit':
            model = timm.create_model("hf-hub:anonauthors/oxford_pet-timm-vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=True)

    # Oxford Flower102
    elif dataset == 'oxford-flowers':
        if model_name == 'resnet':
            model = timm.create_model('hf-hub:anonauthors/flowers102-timm-resnet50', pretrained=True)
        elif model_name == 'vit':
            model = timm.create_model("hf-hub:anonauthors/flowers102-timm-vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=True)

    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    
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
