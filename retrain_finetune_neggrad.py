import hydra
import torch
import os
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from src.utils import get_forgetting_subset
import wandb
from torch.utils.data import Subset
from tqdm import tqdm

from src.datasets.dataset import load_dataset
from src.models.classifier import Classifier
from src.metrics.metrics import compute_metrics
from src.loss.loss import NegGradLoss, NegGradPlusLoss, RandRelabelingLoss
from src.log import get_loggers
from omegaconf import OmegaConf
import multiprocessing


@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg):
    # Set seed
    if cfg.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.seed = seed
    torch.manual_seed(cfg.seed)    

    wandb_logger = get_loggers(cfg) # loggers

    # Load dataset
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    train, val, test = load_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)
    img,lbl = train.__getitem__(0)
    print(img.shape, lbl)
    # retrieving forgetting set for filtering
    forgetting_set = get_forgetting_subset(cfg.forgetting_set, cfg.dataset.classes, cfg.forgetting_set_size)

    model = Classifier(cfg.weights_name, num_classes=cfg[cfg.dataset.name].n_classes, finetune=True)
    model.to(cfg.device)

    optimizer = AdamW(model.parameters(), lr=cfg.train.lr)
    criterion = None    

    if cfg.unlearning_method == 'retrain' or cfg.unlearning_method == 'finetuning':
        criterion = torch.nn.CrossEntropyLoss()
        # Crea una lista di indici che non appartengono al forgetting_set
        idx_train = [i for i, t in enumerate(train.targets) if t not in forgetting_set]
        train = Subset(train, idx_train)

    if cfg.unlearning_method != 'retrain':
        weights = os.path.join(cfg.currentDir, cfg.train.save_path, cfg.dataset.name + '_' + cfg.model + '.pth')
        model.load_state_dict(torch.load(weights, map_location=cfg.device))
    if cfg.unlearning_method == 'neggrad' or cfg.unlearning_method == 'randomlabel':
        idx_train = [i for i, t in enumerate(train.targets) if t in forgetting_set]
        train = Subset(train, idx_train)
        if cfg.unlearning_method == 'neggrad':
            criterion = NegGradLoss()
        elif cfg.unlearning_method == 'randomlabel':
            criterion = RandRelabelingLoss(cfg[cfg.dataset.name].n_classes, forgetting_set)
    if cfg.unlearning_method == 'neggradplus':
        criterion = NegGradPlusLoss(forgetting_set)

    # dataloader of filtered dataset
    train_loader = DataLoader(train, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)
    val_loader = DataLoader(val, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    test_loader = DataLoader(test, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    '''
    # TODO -> DELETE
    model.eval()
    # test
    correct_ret, total_ret, correct_for, total_for = 0,0,0,0
    for i, (x, y) in enumerate(tqdm(test_loader)):
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        y_pred = model(x)
        # softmax
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        _, predicted = y_pred.max(1)
        print(y_pred.shape)

        for label, pred in zip(y, predicted):
            
            print(label.item(), pred.item())
            if label.item() in cfg.forgetting_set:
                # Forget classes
                total_for += 1
                correct_for += (label == pred).item()
            else:
                # Retain classes
                total_ret += 1
                correct_ret += (label == pred).item()
    
    # Compute averages
    retain_acc = 100 * correct_ret / total_ret if total_ret > 0 else 0
    forget_acc = 100 * correct_for / total_for if total_for > 0 else 0
    wandb_logger.log_metrics({"retain_test_acc": retain_acc, "forget_test_acc": forget_acc})
    '''

    # training    
    for epoch in range(cfg.train.max_epochs):
        print(f"Epoch {epoch+1}/{cfg.train.max_epochs}")
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader):
            optimizer.zero_grad()
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        wandb_logger.log_metrics({"train_loss": train_loss})

        # validation
        model.eval()
        with torch.no_grad():
            val_loss, correct_ret, correct_for, total_ret, total_for = 0, 0, 0, 0, 0
            for i, (x, y) in enumerate(tqdm(val_loader)):
                x = x.to(cfg.device)
                y = y.to(cfg.device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
                _, predicted = y_pred.max(1)
                for label, pred in zip(y, predicted):
                    if label.item() in cfg.forgetting_set:
                        total_for += 1
                        correct_for += (label == pred).item()
                    else:
                        total_ret += 1
                        correct_ret += (label == pred).item()

            val_loss /= len(val_loader)
            val_acc = 100 * (correct_ret + correct_for) / (total_ret + total_for)
            wandb_logger.log_metrics({"val_loss": val_loss, "val_acc": val_acc})
            retain_acc = 100 * correct_ret / total_ret if total_ret > 0 else 0
            forget_acc = 100 * correct_for / total_for if total_for > 0 else 0
            wandb_logger.log_metrics({"retain_val_acc": retain_acc, "forget_val_acc": forget_acc})
    
    model.eval()
    # test
    correct_ret, total_ret, correct_for, total_for = 0,0,0,0
    for i, (x, y) in enumerate(tqdm(test_loader)):
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        y_pred = model(x)
        _, predicted = y_pred.max(1)

        for label, pred in zip(y, predicted):
            if label.item() in cfg.forgetting_set:
                # Forget classes
                total_for += 1
                correct_for += (label == pred).item()
            else:
                # Retain classes
                total_ret += 1
                correct_ret += (label == pred).item()
    
    # Compute averages
    retain_acc = 100 * correct_ret / total_ret if total_ret > 0 else 0
    forget_acc = 100 * correct_for / total_for if total_for > 0 else 0
    wandb_logger.log_metrics({"retain_test_acc": retain_acc, "forget_test_acc": forget_acc})

    # save unlearned model
    os.makedirs(os.path.join(cfg.currentDir, cfg.train.save_path), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_forgetting_set_{str(cfg.forgetting_set)}_{cfg.unlearning_method}_{cfg.model}.pth"))
    #metrics=compute_metrics(model, test_loader, cfg.dataset.classes, cfg.forgetting_set)

if __name__ == '__main__':
    print('main')
    main()
