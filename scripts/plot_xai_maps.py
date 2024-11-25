import torch
import hydra
import torchvision
import os
from matplotlib import pyplot as plt

from src.saliency_methods.gradcam import gradcam_interface
from src.models.classifier import Classifier


@hydra.main(config_path='../config', config_name='config')
def main(cfg):
    model = Classifier(
        cfg.weights_name, 
        num_classes=cfg[cfg.dataset.name].n_classes, 
        finetune=True)

    method = gradcam_interface(model, device=cfg.device)

    print(method)

    

    

if __name__ == '__main__':
    main()