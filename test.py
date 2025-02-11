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
    script_dir = hydra.utils.get_original_cwd()
    os.chdir(script_dir)
    
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    _, _, test = load_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)
    test_loader = DataLoader(test, batch_size=32, shuffle=False)

    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    model = Classifier(cfg.weights_name, num_classes=cfg.dataset.classes, finetune=True)
    model.to(cfg.device)

    if cfg.golden_model:
        model.load_state_dict(torch.load('checkpoints/'+cfg.dataset.name+'_'+cfg.model+'_only_retain_set'+str(cfg.forgetting_set)+'.pth'))  # Carica i pesi dal file .pth
    elif cfg.original_model:
        model.load_state_dict(torch.load('checkpoints/'+cfg.dataset.name+'_'+cfg.model+'.pth')) 
    else:
        weights = os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_forgetting_set_{str(cfg.forgetting_set)}_{cfg.unlearning_method}_{cfg.model}.pth")
        model.load_state_dict(torch.load(weights, map_location=cfg.device))
    model.eval()  # evaluation mode

    class_correct = [0] * cfg.dataset.classes  # count of correct predictions for each class
    class_total = [0] * cfg.dataset.classes  # count of images for each class
    total_correct_retain = 0 # count of correct predictions
    total_retain = 0
    total_correct_forget = 0
    total_forget = 0

    with torch.no_grad(): 
        for images, labels in test_loader:
            images, labels = images.to(cfg.device), labels.to(cfg.device)  # data on gpu
            outputs = model(images)  # retrieve predictions
            _, predicted = torch.max(outputs, 1)  # class with highest score

            # counter updating
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
                if label in cfg.forgetting_set:
                    total_forget += 1
                    if predicted[i] == label:
                        total_correct_forget += 1
                else:
                    total_retain += 1
                    if predicted[i] == label:
                        total_correct_retain += 1

    for i in range(cfg.dataset.classes): 
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"Accuracy for class {i}: {accuracy:.2f}%")

    total_correct = sum(class_correct)
    total_images = sum(class_total)
    overall_accuracy = 100 * total_correct / total_images
    print(f'Overall accuracy: {overall_accuracy:.2f}%')
    print(f'Accuracy on retained set: {100 * total_correct_retain / total_retain:.2f}%')
    print(f'Accuracy on forgetting set: {100 * total_correct_forget / total_forget:.2f}%')


if __name__ == '__main__':
    main()
