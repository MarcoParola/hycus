import hydra
import torch
import torchvision
import os
import wandb
import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.classifier import Classifier
from torch.utils.data import DataLoader 
from src.datasets.dataset import load_dataset

def extract_features(model, loader, device):
    model = model.to(device)
    features = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            batch_labels = batch_labels.to(device)
            batch_features = model.extract_features(images)
            features.append(batch_features)
            labels.append(batch_labels)

            # Free gpu memory
            del images, batch_labels, batch_features
            torch.cuda.empty_cache()

    # labels and features are lists of tensors, we concatenate them
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    return features, labels


@hydra.main(config_path='../config', config_name='config', version_base=None)
def main(cfg):
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    train, val, test = load_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)

    # dataloader
    train_loader = torch.utils.data.DataLoader(train, 
        batch_size=cfg.train.batch_size, 
        shuffle=False, 
        num_workers=cfg.train.num_workers)
    
    val_loader = torch.utils.data.DataLoader(val, 
        batch_size=cfg.train.batch_size, 
        shuffle=False, 
        num_workers=cfg.train.num_workers)

    test_loader = torch.utils.data.DataLoader(test, 
        batch_size=cfg.train.batch_size, 
        shuffle=False, 
        num_workers=cfg.train.num_workers)

    model = Classifier(cfg.weights_name, num_classes=cfg[cfg.dataset.name].n_classes, finetune=True)
    if cfg.golden_model==True:
        weights = os.path.join(cfg.currentDir, cfg.train.save_path, cfg.dataset.name + '_resnet_only_retain_set'+str(cfg.forgetting_set)+'.pth')
    else:
        if cfg.original_model==True:
            weights = os.path.join(cfg.currentDir, cfg.train.save_path, cfg.dataset.name + '_resnet.pth')
        else:
            if cfg.golden_model==False:
                weights = os.path.join(cfg.currentDir, cfg.train.save_path, cfg.dataset.name + '_forgetting_set_'+str(cfg.forgetting_set)+'_' +cfg.unlearning_method+'_'+ cfg.model + '.pth')
            else:
                weights = os.path.join(cfg.currentDir, cfg.train.save_path, cfg.dataset.name + '_resnet_only_retain_set'+str(cfg.forgetting_set)+'.pth')
    
    model.load_state_dict(torch.load(weights, map_location=cfg.device))
    torch.grad = False

    # train features extraction
    features, labels = extract_features(model, train_loader, cfg.device)
    if cfg.golden_model==True:
        torch.save(features, f"data/features/{cfg.dataset.name}/train_features_only_retain_forgetting_{cfg.forgetting_set}.pt")
        torch.save(labels, f"data/features/{cfg.dataset.name}/train_labels_only_retain_forgetting_{cfg.forgetting_set}.pt")
    else:
        if cfg.original_model==True:
            torch.save(features, f"data/features/{cfg.dataset.name}/train_features_original.pt")
            torch.save(labels, f"data/features/{cfg.dataset.name}/train_labels_original.pt")
        else:
            torch.save(features, f"data/features/{cfg.dataset.name}/train_features_{cfg.unlearning_method}_{cfg.forgetting_set}.pt")
            torch.save(labels, f"data/features/{cfg.dataset.name}/train_labels_{cfg.unlearning_method}_{cfg.forgetting_set}.pt")
    
    # validation features extraction
    features, labels = extract_features(model, val_loader, cfg.device)
    if cfg.original_model==True:
        torch.save(features, f"data/features/{cfg.dataset.name}/val_features_original.pt")
        torch.save(labels, f"data/features/{cfg.dataset.name}/val_labels_original.pt")
    else:
        if cfg.golden_model==True:
            torch.save(features, f"data/features/{cfg.dataset.name}/val_features_only_retain_forgetting_{cfg.forgetting_set}.pt")
            torch.save(labels, f"data/features/{cfg.dataset.name}/val_labels_only_retain_forgetting_{cfg.forgetting_set}.pt")
        else:
            torch.save(features, f"data/features/{cfg.dataset.name}/val_features_{cfg.unlearning_method}_{cfg.forgetting_set}.pt")
            torch.save(labels, f"data/features/{cfg.dataset.name}/val_labels_{cfg.unlearning_method}_{cfg.forgetting_set}.pt")

    # test features extraction
    features, labels = extract_features(model, test_loader, cfg.device)
    if cfg.golden_model==True:
        torch.save(features, f"data/features/{cfg.dataset.name}/test_features_only_retain_forgetting_{cfg.forgetting_set}.pt")
        torch.save(labels, f"data/features/{cfg.dataset.name}/test_labels_only_retain_forgetting_{cfg.forgetting_set}.pt")
    else:
        if cfg.original_model==True:
            torch.save(features, f"data/features/{cfg.dataset.name}/test_features_original.pt")
            torch.save(labels, f"data/features/{cfg.dataset.name}/test_labels_original.pt")
        else:
            torch.save(features, f"data/features/{cfg.dataset.name}/test_features_{cfg.unlearning_method}_{cfg.forgetting_set}.pt")
            torch.save(labels, f"data/features/{cfg.dataset.name}/test_labels_{cfg.unlearning_method}_{cfg.forgetting_set}.pt")
    
if __name__ == '__main__':
    main()
