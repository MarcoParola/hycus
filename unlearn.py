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
from src.unlearning.factory import get_unlearning_method
from omegaconf import OmegaConf
from src.utils import get_retain_and_forget_datasets
from src.dataset_wrapper import DatasetWrapper
from src.dataset_wrapper_icus import DatasetWrapperIcus
from src.models.resnet import ResNet, ResidualBlock
from src.unlearning_methods.icus import Icus


@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg):
    # Set seed
    if cfg.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.seed = seed
    torch.manual_seed(cfg.seed)    

    # loggers
    loggers = get_loggers(cfg)

    # Load dataset
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    train, val, test = load_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)
    # TODO fai il wrap con la classe custom: ImgTextDataset
    test_loader = torch.utils.data.DataLoader(test, 
        batch_size=cfg.train.batch_size, 
        shuffle=False, 
        num_workers=cfg.train.num_workers)

    train_loader = torch.utils.data.DataLoader(train, 
        batch_size=cfg.train.batch_size, 
        shuffle=False, 
        num_workers=cfg.train.num_workers)

    # Load model
    print("Carico il modello")
    #model = load_model(cfg.model, cfg.dataset.name)
    model=ResNet(ResidualBlock)
    model.load_state_dict(torch.load(os.path.join(cfg.currentDir, "checkpoints", cfg.dataset.name + '_' + cfg.model + '.pth'), map_location=cfg.device))
    model.to(cfg.device)

    print("Calcolo le metriche")
    # compute classification metrics
    num_classes = cfg[cfg.dataset.name].n_classes
    forgetting_subset = get_forgetting_subset(cfg.forgetting_set, cfg[cfg.dataset.name].n_classes, cfg.forgetting_set_size)
    metrics = compute_metrics(model, test_loader, num_classes, forgetting_subset)
    for k, v in metrics.items():
        print(f'{k}: {v}')

    print("Wrapper datasets")
    retain_dataset, forget_dataset, forget_indices = get_retain_and_forget_datasets(train, forgetting_subset, cfg.forgetting_set_size)
    #print("Indici da scordare:", forget_indices)
    retain_indices = [i for i in range(len(train)) if i not in forget_indices]
    # unlearning process
    unlearning_method = cfg.unlearning_method
    
    if unlearning_method == 'icus':
        infgt = np.zeros(num_classes)
        for i in forgetting_subset:
            infgt[i] = 1
        cuda = True if cfg.device == 'cuda' else False
        wrapped_train = DatasetWrapperIcus(infgt, model, cuda, orig_dataset=cfg.dataset.name)
        wrapped_train_loader = DataLoader(wrapped_train, batch_size=10, num_workers=0) #SHUFFLE ?????
        unlearning = Icus(cfg, model, 128, num_classes, wrapped_train_loader, forgetting_subset) #128 perchè CIFAR10 probabilmente serve una logica per trovare il valore    
    else:
        wrapped_train = DatasetWrapper(train, forget_indices)
        retain_loader = DataLoader(wrapped_train, batch_size=cfg.train.batch_size,sampler=SubsetRandomSampler(retain_indices), num_workers=8) 
        forget_loader = DataLoader(wrapped_train, batch_size=cfg.train.batch_size, sampler=SubsetRandomSampler(forget_indices), num_workers=8)
        wrapped_train_loader = DataLoader(wrapped_train, batch_size=cfg.train.batch_size, shuffle=True, num_workers=8)
        unlearning = get_unlearning_method(unlearning_method, model, retain_loader, forget_loader, test_loader, train_loader, cfg, forgetting_subset)
    if unlearning_method == 'scrub':
        unlearning.unlearn(wrapped_train_loader)
    elif unlearning_method == 'badT':
        unlearning.unlearn(wrapped_train_loader, test_loader)
    elif unlearning_method == 'ssd':
        unlearning.unlearn(wrapped_train_loader, test_loader, forget_loader)
    elif unlearning_method == 'icus':
        new_model=unlearning.unlearn(model, wrapped_train_loader)
    else:
        raise ValueError(f"Unlearning method '{unlearning_method}' not recognised.")
    # recompute metrics
    if unlearning_method == 'icus':
        accuracy_retain, accuracy_forget=unlearning.test_unlearning_effect(test_loader, forgetting_subset)
        loggers.log_metrics({'accuracy_retain': accuracy_retain, 'accuracy_forget': accuracy_forget})
    else:
        metrics = compute_metrics(unlearning.model, test_loader, num_classes, forgetting_subset)
        print("Accuracy forget ", metrics['accuracy_forgetting'])
        print("Accuracy retain ", metrics['accuracy_retaining'])



if __name__ == '__main__':
    main()