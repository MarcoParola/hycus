import hydra
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

#from src.utils import get_early_stopping, get_save_model_callback
from src.models.model import load_model
from src.datasets.dataset import load_dataset
from src.metrics.metrics import compute_metrics
from src.log import get_loggers
from src.utils import get_forgetting_subset
from src.unlearning_methods.base import get_unlearning_method
from src.utils import get_retain_and_forget_datasets
from src.dataset_wrapper import DatasetWrapper
from src.dataset_wrapper_icus import DatasetWrapperIcus
from src.models.resnet import ResNet, ResidualBlock # TODO remove this import
from src.metrics.metrics import get_membership_attack_prob, compute_mia, prepare_membership_inference_attack
from src.unlearning_methods.icus import Icus


@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg):
    # Set seed
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
    # use load_model function defined in src/models/model.py
    # remember to fix that function
    model = load_model(cfg.model, os.path.join(cfg.currentDir, cfg.train.save_path, cfg.dataset.name + '_' + cfg.model + '.pth') , cfg.device)

    print("Compute classification metrics")
    num_classes = cfg[cfg.dataset.name].n_classes
    forgetting_subset = get_forgetting_subset(cfg.forgetting_set, cfg[cfg.dataset.name].n_classes, cfg.forgetting_set_size)
    metrics = compute_metrics(model, test_loader, num_classes, forgetting_subset)
    for k, v in metrics.items():
        print(f'{k}: {v}')

    print("Wrapping datasets")
    retain_dataset, forget_dataset, forget_indices = get_retain_and_forget_datasets(train, forgetting_subset, cfg.forgetting_set_size)
    forget_indices_val = [i for i in range(len(val)) if val[i][1] in forgetting_subset]
    retain_indices = [i for i in range(len(train)) if i not in forget_indices]
    
    # unlearning process
    unlearning_method = cfg.unlearning_method
    wrapped_val = DatasetWrapper(val, forget_indices_val)
    wrapped_val_loader = DataLoader(wrapped_val, batch_size=cfg.train.batch_size, shuffle=True, num_workers=8)
    retain_loader = DataLoader(retain_dataset, batch_size=cfg.train.batch_size, num_workers=8) 
    forget_loader = DataLoader(forget_dataset, batch_size=cfg.train.batch_size, num_workers=8)
    if unlearning_method == 'icus':
        infgt = torch.zeros(num_classes)
        infgt[forgetting_subset] = 1
        wrapped_train = DatasetWrapperIcus(infgt, model, num_classes=num_classes, dataset_name=cfg.dataset.name, device=cfg.device)
        wrapped_train_loader = DataLoader(wrapped_train, batch_size=10, num_workers=0) #SHUFFLE ?????
        unlearning = Icus(cfg, model, 128, num_classes, wrapped_train_loader, forgetting_subset, wrapped_val_loader, loggers) #128 because we're using ResNet, we probably need a logic to get the number of features    
    else:
        wrapped_train = DatasetWrapper(train, forget_indices)
        wrapped_train_loader = DataLoader(wrapped_train, batch_size=cfg.train.batch_size*2, shuffle=True, num_workers=8)
        forget_loader_with_infgt = DataLoader(wrapped_train, batch_size=cfg.train.batch_size//4, sampler=SubsetRandomSampler(forget_indices))
        retain_loader_with_infgt = DataLoader(wrapped_train, batch_size=cfg.train.batch_size, sampler=SubsetRandomSampler(retain_indices))
        unlearning = get_unlearning_method(unlearning_method, model, test_loader, train_loader, wrapped_val_loader, cfg, forgetting_subset, loggers)
    if unlearning_method == 'scrub':
        new_model = unlearning.unlearn(wrapped_train_loader, forget_loader_with_infgt)
    elif unlearning_method == 'badT':
        unlearning.unlearn(wrapped_train_loader, test_loader, wrapped_val_loader)
    elif unlearning_method == 'ssd':
        unlearning.unlearn(wrapped_train_loader, test_loader, forget_loader)
    elif unlearning_method == 'icus':
        new_model=unlearning.unlearn(model, wrapped_train_loader)
    else:
        raise ValueError(f"Unlearning method '{unlearning_method}' not recognised.")
    # recompute metrics
    if unlearning_method == 'icus':
        unlearning.test_unlearning_effect(wrapped_train_loader, test_loader, forgetting_subset, epoch=cfg.max_epochs, test=True)
    else:
        metrics = compute_metrics(unlearning.model, test_loader, num_classes, forgetting_subset)
        print("Accuracy forget ", metrics['accuracy_forgetting'])
        print("Accuracy retain ", metrics['accuracy_retaining'])
    test_forget_loader= prepare_membership_inference_attack(test, forgetting_subset, cfg.train.batch_size) 
    compute_mia(test_forget_loader, forget_loader, new_model, num_classes, loggers)
    #compute_mia(retain_loader, forget_loader, test_loader, new_model, num_classes, forgetting_subset, loggers)



if __name__ == '__main__':
    main()