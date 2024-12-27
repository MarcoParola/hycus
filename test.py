import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from src.datasets.dataset import load_dataset
from src.models.classifier import Classifier
from scripts.extract_features import extract_features
from scripts.plot.confusion_matrix import compute_confusion_matrix
import torch.optim as optim
import numpy as np
import os
import hydra


@hydra.main(config_path='config', config_name='config')
def main(cfg):
    print(f"Current directory before adjustment: {os.getcwd()}")
    # Se necessario, puoi tornare alla directory originale
    script_dir = hydra.utils.get_original_cwd()
    os.chdir(script_dir)
    print(f"Directory corrente ripristinata a: {os.getcwd()}")

    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    _, _, test = load_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)
    test_loader = DataLoader(test, batch_size=32, shuffle=False)

    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    model = Classifier(cfg.weights_name, num_classes=cfg[cfg.dataset.name].n_classes, finetune=True)
    model.to(cfg.device)

    if cfg.golden_model:
        model.load_state_dict(torch.load('checkpoints/'+cfg.dataset.name+'_'+cfg.model+'_only_retain_set'+str(cfg.forgetting_set)+'.pth'))  # Carica i pesi dal file .pth
    else:
        weights = os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_{cfg.model}.pth")
        model.load_state_dict(torch.load(weights, map_location=cfg.device))
    model.eval()  # Imposta il modello in modalità di valutazione

    class_correct = [0] * cfg.dataset.classes  # Contatore delle previsioni corrette per ciascuna classe (10 classi in CIFAR-10)
    class_total = [0] * cfg.dataset.classes  # Contatore delle immagini totali per ciascuna classe

    with torch.no_grad():  # Disattiva il calcolo dei gradienti
        for images, labels in test_loader:
            images, labels = images.to(cfg.device), labels.to(cfg.device)  # Sposta i dati su GPU
            outputs = model(images)  # Ottieni le previsioni del modello
            _, predicted = torch.max(outputs, 1)  # Ottieni la classe con la probabilità più alta

            # Aggiorna il contatore per ogni classe
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    for i in range(cfg.dataset.classes): 
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"Accuracy for class {i}: {accuracy:.2f}%")

    total_correct = sum(class_correct)
    total_images = sum(class_total)
    overall_accuracy = 100 * total_correct / total_images
    print(f'Overall accuracy: {overall_accuracy:.2f}%')

    features, labels = extract_features(model, test_loader, cfg.device)
    torch.save(features, f"data/features/"+cfg.dataset.name+"/only_retain_set_features"+str(cfg.forgetting_set)+".pt")

    compute_confusion_matrix(model, test_loader, cfg)    


if __name__ == '__main__':
    print('main')
    main()
