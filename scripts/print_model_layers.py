import torch
import hydra
import torchvision
import os
import sys

from torch import nn
from torchvision import models
from torchvision.models import ResNet34_Weights, resnet34, VGG11_Weights, vgg11
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.classifier import Classifier


@hydra.main(config_path='../config', config_name='config')
def main(cfg):
    model = Classifier(
        weights=cfg.weights_name,
        num_classes=cfg[cfg.dataset.name].n_classes,
        #lr=cfg.train.lr,
        #max_epochs=cfg.train.max_epochs,
        finetune=True
    )

    for name, module in model.named_modules():
        print(name)
        print(sum(p.numel() for p in module.parameters() if p.requires_grad))

if __name__ == '__main__':
    main()
