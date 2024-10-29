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
from src.datasets.unlearning_dataset import get_unlearning_datasets
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
    model = load_model(cfg.model, os.path.join(cfg.currentDir, cfg.train.save_path, cfg.dataset.name + '_' + cfg.model + '.pth') , cfg.device)

    print("Compute classification metrics")
    num_classes = cfg[cfg.dataset.name].n_classes
    forgetting_subset = get_forgetting_subset(cfg.forgetting_set, cfg[cfg.dataset.name].n_classes, cfg.forgetting_set_size)
    metrics = compute_metrics(model, test_loader, num_classes, forgetting_subset)
    for k, v in metrics.items():
        print(f'{k}: {v}')

    # prepare datasets for unlearning
    print("Wrapping datasets")
    retain_dataset, forget_dataset, forget_indices = get_retain_and_forget_datasets(train, forgetting_subset, cfg.forgetting_set_size)
    forget_indices_val = [i for i in range(len(val)) if val[i][1] in forgetting_subset]
    retain_indices = [i for i in range(len(train)) if i not in forget_indices]
    
    unlearning_train = get_unlearning_datasets(train, retain_indices, forget_indices, cfg)
    retain_loader = DataLoader(retain_dataset, batch_size=cfg.train.batch_size, num_workers=8) 
    forget_loader = DataLoader(forget_dataset, batch_size=cfg.train.batch_size, num_workers=8)

    # start unlearning process
    unlearning_method_name = cfg.unlearning_method
    unlearning_method = get_unlearning_method(unlearning_method_name, model, test_loader, train_loader, wrapped_val_loader, cfg, forgetting_subset, loggers)
    new_model = None
    if unlearning_method_name == 'icus':
        new_model = unlearning_method.unlearn() # TODO
    elif unlearning_method_name == 'scrub':
        new_model = unlearning_method.unlearn() # TODO
    elif unlearning_method_name == 'badT':
        new_model = unlearning_method.unlearn() # TODO
    elif unlearning_method_name == 'ssd':
        new_model = unlearning_method.unlearn() # TODO

    metrics = compute_metrics(new_model, test_loader, num_classes, forgetting_subset)
    print("Accuracy forget ", metrics['accuracy_forgetting'])
    print("Accuracy retain ", metrics['accuracy_retaining'])



if __name__ == '__main__':
    main()