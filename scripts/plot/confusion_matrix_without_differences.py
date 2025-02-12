import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import hydra
import torchvision
from datetime import datetime
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.datasets.dataset import load_dataset
from src.models.classifier import Classifier
from scripts.plot.confusion_matrix import compute_confusion_matrix
from matplotlib.colors import TwoSlopeNorm, Normalize

def plot_all_the_confusion_matrices(cms, cfg, names, labels=None, rows=1, cols=6, positions=None):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()  

    if positions is None:
        positions = list(range(len(cms)))

    # Find max and min values for normalization
    overall_min = min(cm.min() for cm in cms)
    overall_max = max(cm.max() for cm in cms)

    # normalization
    norm = mcolors.Normalize(vmin=overall_min, vmax=overall_max)

    for i, pos in enumerate(positions):
        if i < len(cms) and pos < len(axes):
            ax = axes[pos]
            im = ax.imshow(cms[i], interpolation='nearest', cmap=plt.cm.Blues, norm=norm)
            ax.set_title(f"Matrix {names[i]}")
            if labels is not None:
                ax.set_xticks(np.arange(len(labels)))
                ax.set_yticks(np.arange(len(labels)))
                ax.set_xticklabels(labels)
                ax.set_yticklabels(labels)
            if cfg.dataset.name=='cifar10':
                for j in range(cms[i].shape[0]):
                    for k in range(cms[i].shape[1]):
                        ax.text(k, j, f"{cms[i][j, k]}", ha="center", va="center", color="black")

    for j in range(len(cms), len(axes)):
        axes[j].axis('off')

    # Salva la figura
    plt.savefig("src/data/all_the_confusion_matrices_" + str(cfg.forgetting_set) + ".pdf")
    

@hydra.main(config_path='../../config', config_name='config', version_base=None)
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
    #original
    weights= os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_{cfg.model}.pth")
    model.load_state_dict(torch.load(weights, map_location=cfg.device))
    cm1 = compute_confusion_matrix(model, test_loader, cfg, save_plot=False)
    cms.append(cm1)

    #golden
    weights=os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_{cfg.model}_only_retain_set{cfg.forgetting_set}.pth")
    model.load_state_dict(torch.load(weights, map_location=cfg.device))
    cm2 = compute_confusion_matrix(model, test_loader, cfg, save_plot=False)
    cms.append(cm2)
    
    #scrub
    weights=os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_forgetting_set_{cfg.forgetting_set}_scrub_{cfg.model}.pth")
    model.load_state_dict(torch.load(weights, map_location=cfg.device))
    cm3 = compute_confusion_matrix(model, test_loader, cfg, save_plot=False)
    cms.append(cm3)

    #ssd
    weights=os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_forgetting_set_{cfg.forgetting_set}_ssd_{cfg.model}.pth")
    model.load_state_dict(torch.load(weights, map_location=cfg.device))
    cm4 = compute_confusion_matrix(model, test_loader, cfg, save_plot=False)
    cms.append(cm4)

    #badT
    weights=os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_forgetting_set_{cfg.forgetting_set}_badT_{cfg.model}.pth")
    model.load_state_dict(torch.load(weights, map_location=cfg.device))
    cm5 = compute_confusion_matrix(model, test_loader, cfg, save_plot=False)
    cms.append(cm5)

    #icus
    weights=os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_forgetting_set_{cfg.forgetting_set}_icus_{cfg.model}.pth")
    model.load_state_dict(torch.load(weights, map_location=cfg.device))
    cm6 = compute_confusion_matrix(model, test_loader, cfg, save_plot=False)
    cms.append(cm6)

    #finetuning
    weights=os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_forgetting_set_{cfg.forgetting_set}_finetuning_{cfg.model}.pth")
    model.load_state_dict(torch.load(weights, map_location=cfg.device))
    cm7 = compute_confusion_matrix(model, test_loader, cfg, save_plot=False)
    cms.append(cm7)

    names=["original", "golden", "scrub", "ssd", "badT", "icus", "finetuning"]

    plot_all_the_confusion_matrices(cms, cfg, names, labels=np.arange(cfg.dataset.classes), rows=1, cols=6)

if __name__ == "__main__":
    main()