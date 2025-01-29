import hydra
import torch
import os
import numpy as np
import copy
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from src.models.model import load_model
from src.datasets.dataset import load_dataset, get_retain_forget_dataloaders
from src.metrics.metrics import compute_metrics, add_case, update_case
from src.log import get_loggers
from src.utils import get_forgetting_subset
from torch.utils.data import Subset
from scripts.plot.pca_tsne import plot_features, plot_features_3d
from src.unlearning_methods.base import get_unlearning_method
from src.utils import get_retain_and_forget_datasets
from src.datasets.unlearning_dataset import UnlearningDataset
from src.datasets.unlearning_dataset import get_unlearning_dataset
from src.models.resnet import ResNet9, ResNet18, ResidualBlock 
from src.models.classifier import Classifier
from src.unlearning_methods.icus import Icus, IcusHierarchy



@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg):
    # Set seed
    torch.cuda.empty_cache() 
    if cfg.seed == -1:
        random_data = os.urandom(4)
        print("Random classes: ", random_data)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.seed = seed
    torch.manual_seed(cfg.seed)    

    # loggers
    loggers = get_loggers(cfg)

    # Load dataset
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    train, val, test = load_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)
    
    # Data loaders
    test_loader = torch.utils.data.DataLoader(test, 
        batch_size=cfg.train.batch_size, 
        shuffle=False, 
        num_workers=cfg.train.num_workers)

    train_loader = torch.utils.data.DataLoader(train, 
        batch_size=cfg.train.batch_size, 
        shuffle=False, 
        num_workers=cfg.train.num_workers)
    
    val_loader = torch.utils.data.DataLoader(val, 
        batch_size=cfg.train.batch_size, 
        shuffle=False, 
        num_workers=cfg.train.num_workers)

    # Load model
    print("Model loading")
    model = Classifier(cfg.weights_name, num_classes=cfg[cfg.dataset.name].n_classes, finetune=True)
    model.to(cfg.device)
    # load model weights
    weights = os.path.join(cfg.currentDir, cfg.train.save_path, cfg.dataset.name + '_' + cfg.model + '.pth')
    model.load_state_dict(torch.load(weights, map_location=cfg.device))
    print("Compute classification metrics")
    num_classes = cfg.dataset.classes
    #Forgetting set loading
    forgetting_subset = get_forgetting_subset(cfg.forgetting_set, cfg.dataset.classes, cfg.forgetting_set_size)
    # Compute classification metrics of the original model
    metrics = compute_metrics(model, test_loader, num_classes, forgetting_subset)
    for k, v in metrics.items():
        print(f'{k}: {v}')

    # Plotting
    pca, shared_limits = plot_features(model, test_loader, forgetting_subset, unlearned=False)
    pca=plot_features_3d(model, test_loader, forgetting_subset)
    
    #prepare datasets for unlearning
    print("Wrapping datasets")
    retain_dataset, forget_dataset, forget_indices = get_retain_and_forget_datasets(train, forgetting_subset, cfg.forgetting_set_size)
    print("Forget indices: ", len(forget_indices))
    forget_indices_val = [i for i in range(len(val)) if val[i][1] in forgetting_subset]
    retain_indices = [i for i in range(len(train)) if i not in forget_indices]
    
    unlearning_method_name = cfg.unlearning_method
    unlearning_train = get_unlearning_dataset(cfg, unlearning_method_name, model, train, retain_indices, forget_indices, forgetting_subset)
    
    #retain and forget set loaders
    retain_loader, forget_loader = get_retain_forget_dataloaders(cfg, retain_dataset, forget_dataset)
    
    # unlearning process
    unlearning_method = get_unlearning_method(cfg, unlearning_method_name, model, unlearning_train, forgetting_subset, loggers)
    new_model = None
    if unlearning_method_name == 'icus':
        new_model = unlearning_method.unlearn(model, unlearning_train, val_loader)
    if unlearning_method_name == 'icus_hierarchy':
        new_model = unlearning_method.unlearn(model, unlearning_train, val_loader)
    elif unlearning_method_name == 'scrub':
        new_model = unlearning_method.unlearn(retain_loader, forget_loader, val_loader) 
    elif unlearning_method_name == 'badT':
        new_model = unlearning_method.unlearn(unlearning_train, test_loader, val_loader) 
    elif unlearning_method_name == 'ssd':
        if cfg.unlearn.already_forgotten_classes != []:
            weights = os.path.join(cfg.currentDir, cfg.train.save_path, cfg.dataset.name + '_forgetting_set_'+str(cfg.unlearn.already_forgotten_classes)+'_ssd_' + cfg.model + '.pth')
            model.load_state_dict(torch.load(weights, map_location=cfg.device))
        
        labels = np.array(train.targets)
        indices = np.where(~np.isin(labels, cfg.unlearn.already_forgotten_classes))[0]
        filtered_train = Subset(train, indices)
        retain_dataset, forget_dataset, forget_indices = get_retain_and_forget_datasets(filtered_train, cfg.forgetting_set, 1)
        unlearning_train = UnlearningDataset(filtered_train, forget_indices)
        unlearning_train = torch.utils.data.DataLoader(unlearning_train, batch_size=cfg.train.batch_size, num_workers=8)
        retain_loader, forget_loader = get_retain_forget_dataloaders(cfg, retain_dataset, forget_dataset)
        new_model = unlearning_method.unlearn(model, unlearning_train, test_loader, forget_loader)
        forgetting_subset.extend(cfg.unlearn.already_forgotten_classes) 
        print(forgetting_subset)
    
    plot_features(new_model, test_loader, forgetting_subset, pca=pca, unlearned=True, shared_limits=shared_limits)
    plot_features_3d(new_model, test_loader, forgetting_subset, pca, True)
    
    # Save new model
    os.makedirs(os.path.join(cfg.currentDir, cfg.train.save_path), exist_ok=True)
    if cfg.unlearning_method == 'ssd':
        torch.save(new_model.state_dict(), os.path.join(cfg.currentDir, cfg.train.save_path, cfg.dataset.name + '_forgetting_set_' + str(forgetting_subset) +'_'+cfg.unlearning_method+'_' + cfg.model + '.pth'))    
    else:
        if cfg.unlearning_method != 'icus':
            torch.save(new_model.state_dict(), os.path.join(cfg.currentDir, cfg.train.save_path, cfg.dataset.name + '_forgetting_set_' + str(cfg.forgetting_set) +'_'+cfg.unlearning_method+'_'+ cfg.model + '.pth'))
        if cfg.unlearning_method == 'icus' and cfg.unlearn.reconstruct_from_d == False:
            torch.save(new_model.state_dict(), os.path.join(cfg.currentDir, cfg.train.save_path, cfg.dataset.name + '_forgetting_set_' + str(cfg.forgetting_set) +'_'+cfg.unlearning_method+'_' +cfg.unlearn.aggregation_method+ '_'  + cfg.model + '.pth'))
    
    # log metrics
    metrics = compute_metrics(new_model, test_loader, num_classes, forgetting_subset)
    loggers.log_metrics({
            "retain_test_accuracy": metrics['accuracy_retaining'],
            "forget_test_accuracy": metrics['accuracy_forgetting'],
            "step": 0
        })
    print("Accuracy forget ", metrics['accuracy_forgetting'])
    print("Accuracy retain ", metrics['accuracy_retaining'])

    if cfg.unlearn.update_json == True:
        with open("src/metrics/metrics.json", "r") as file:
            data = json.load(file)
        done = add_case(data, cfg.unlearning_method, str(cfg.forgetting_set), metrics['accuracy_retaining'], metrics['accuracy_forgetting'])
        if not done:
            done = update_case(data, cfg.unlearning_method, str(cfg.forgetting_set), metrics['accuracy_retaining'], metrics['accuracy_forgetting'])
        if not done:
            print("Failed to add/update json")

if __name__ == '__main__':
    main()
