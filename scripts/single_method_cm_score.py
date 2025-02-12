import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import hydra
import torchvision
from datetime import datetime
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.datasets.dataset import load_dataset
from src.models.classifier import Classifier
from scripts.descr_and_similarity import calculate_embeddings, calculate_dissimilarity
from matplotlib.colors import TwoSlopeNorm
from scripts.plot.confusion_matrix import compute_confusion_matrix, difference_between_matrices, calculate_cm_error, calculate_weighted_cm_error



@hydra.main(config_path='../config', config_name='config', version_base=None)
def main(cfg):
    # Percorso dei dati
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((224, 224)),  # Ridimensionamento delle immagini
    ])

    # Carica il dataset CIFAR-10
    if cfg.dataset.name == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    elif cfg.dataset.name == 'cifar100':
        test_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)

    # Crea il DataLoader per il test
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

    # Carica il modello
    model = Classifier(cfg.weights_name, num_classes=cfg[cfg.dataset.name].n_classes, finetune=True)
    model.to(cfg.device)

    cms=[]
    cm1=cm2=cm3=None
    names=[]
    #golden
    weights=os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_{cfg.model}_only_retain_set{cfg.forgetting_set}.pth")
    if os.path.exists(weights):
        names.append("original")
        model.load_state_dict(torch.load(weights, map_location=cfg.device))
        cm1 = compute_confusion_matrix(model, test_loader, cfg, save_plot=False)
        cms.append(cm1)
    if cfg.unlearning_method == "icus":
        weights = os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_forgetting_set_{cfg.forgetting_set}_{cfg.unlearning_method}_{cfg.unlearn.aggregation_method}_{cfg.model}.pth")
    else:
        unlearned_weights = os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_forgetting_set_{cfg.forgetting_set}_{cfg.unlearning_method}_{cfg.model}.pth")
    model.load_state_dict(torch.load(unlearned_weights, map_location=cfg.device))

    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location=cfg.device))
        cm2 = compute_confusion_matrix(model, test_loader, cfg, save_plot=False)
        cms.append(cm2)
        if cfg.unlearning_method != "icus":
            names.append(cfg.unlearning_method)
        else:
            names.append(cfg.unlearning_method+"_"+cfg.unlearn.aggregation_method)

    embeddings = calculate_embeddings(cfg.dataset.name)
    embeddings = embeddings.mean(dim=1)
    embeddings_dissimilarity = calculate_dissimilarity(embeddings)

    cm3=difference_between_matrices(cm1, cm2)

    print("Errore non pesato "+cfg.unlearning_method+" : ", calculate_cm_error(test_loader, cm3, cfg.dataset.classes))
    print("Errore pesato "+cfg.unlearning_method+" : ", calculate_weighted_cm_error(test_loader, cm3, embeddings_dissimilarity, cfg.dataset.classes))

if __name__ == "__main__":
    main()