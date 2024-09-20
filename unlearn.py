import hydra
import torch
import os

#from src.utils import get_early_stopping, get_save_model_callback
from src.models.model import load_model
from src.datasets.dataset import load_dataset
from src.metrics.metrics import compute_metrics
from src.log import get_loggers
from src.utils import get_forgetting_subset
from omegaconf import OmegaConf



@hydra.main(config_path='config', config_name='config')
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

    # Load model
    model = load_model(cfg.model, cfg.dataset.name)
    model.to(cfg.device)

    # compute classification metrics
    num_classes = cfg[cfg.dataset.name].n_classes
    forgetting_subset = get_forgetting_subset(cfg.forgetting_set, cfg[cfg.dataset.name].n_classes, cfg.forgetting_set_size)
    metrics = compute_metrics(model, test_loader, num_classes, forgetting_subset)
    for k, v in metrics.items():
        print(f'{k}: {v}')

    # unlearning process
    unlearning_method = None

    # recompute metrics




if __name__ == '__main__':
    main()