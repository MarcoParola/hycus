import hydra
import torch
import os

#from src.utils import get_early_stopping, get_save_model_callback
from src.models.model import load_model
from src.datasets.dataset import load_dataset
from src.metrics.metrics import compute_classification_metrics
from src.log import get_loggers



@hydra.main(config_path='config', config_name='config')
def main(cfg):

    # Set seed
    if cfg.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.seed = seed
    torch.manual_seed(cfg.seed)


    # callback
    '''
    callbacks = list()
    callbacks.append(get_early_stopping(cfg.train.patience))
    finetune = "finetuned_" if cfg.train.finetune else "no_finetuned_"
    model_save_dir = os.path.join(cfg.currentDir, cfg.checkpoint, finetune + cfg.model + cfg.dataset.name )
    callbacks.append(get_save_model_callback(model_save_dir))
    '''


    # loggers
    loggers = get_loggers(cfg)

    # Load dataset
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    train, val, test = load_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)
    train_loader = torch.utils.data.DataLoader(train, 
        batch_size=cfg.train.batch_size, 
        shuffle=True, 
        num_workers=cfg.train.num_workers)
    val_loader = torch.utils.data.DataLoader(val, 
        batch_size=cfg.train.batch_size, 
        shuffle=False, 
        num_workers=cfg.train.num_workers)
    test_loader = torch.utils.data.DataLoader(test, 
        batch_size=cfg.train.batch_size, 
        shuffle=False, 
        num_workers=cfg.train.num_workers)

    # Load model
    model = load_model(cfg.model, cfg.dataset.name)
    model.to(cfg.device)

    # compute classification metrics
    num_classes = cfg[cfg.dataset.name].n_classes
    forgetting_subset = [1,4,8]
    metrics = compute_classification_metrics(model, test_loader, num_classes, forgetting_subset)
    # pretty dictionary display
    for k, v in metrics.items():
        print(f'{k}: {v}')




if __name__ == '__main__':
    main()