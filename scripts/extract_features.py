import hydra
import torch
import torchvision
import os
import wandb
import tqdm

from src.models.classifier import Classifier
from src.datasets.dataset import load_dataset

def extract_features(model, dataset, device):
    features = []
    labels = []

    for image, label in tqdm(dataset):
        image = image.unsqueeze(0)
        image = image.to(device)
        features.append(model.extract_features(image))
        labels.append(label)

    return features, labels

@hydra.main(config_path='../config', config_name='config')
def main(cfg):
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    train, val, test = load_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)

    model = Classifier(cfg.weights_name, num_classes=cfg[cfg.dataset.name].n_classes, finetune=True)

    # extract featiures from train, val and test and save them 
    # in three different files on ./data/features/{datasaet_name}/

    # extract features from train
    features, labels = extract_features(model, train, cfg.device)
    torch.save(features, f"./data/features/{cfg.dataset.name}/train_features.pt")
    torch.save(labels, f"./data/features/{cfg.dataset.name}/train_labels.pt")

    # extract features from val
    features, labels = extract_features(model, val, cfg.device)
    torch.save(features, f"./data/features/{cfg.dataset.name}/val_features.pt")
    torch.save(labels, f"./data/features/{cfg.dataset.name}/val_labels.pt")

    # extract features from test
    features, labels = extract_features(model, test, cfg.device)
    torch.save(features, f"./data/features/{cfg.dataset.name}/test_features.pt")
    torch.save(labels, f"./data/features/{cfg.dataset.name}/test_labels.pt")
    