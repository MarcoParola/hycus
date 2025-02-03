import hydra
import torch
import os
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from src.datasets.dataset import load_dataset
from src.models.classifier import Classifier
from src.log import get_loggers
from omegaconf import OmegaConf

@hydra.main(config_path='config', config_name='config')
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
    train_loader = DataLoader(train, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)
    val_loader = DataLoader(val, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    test_loader = DataLoader(test, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)


    model = Classifier(cfg.weights_name, num_classes=cfg[cfg.dataset.name].n_classes, finetune=True)
    model.to(cfg.device)

    optimizer = AdamW(model.parameters(), lr=cfg.train.lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(cfg.train.max_epochs):
        print(f"Epoch {epoch+1}/{cfg.train.max_epochs}")
        model.train()
        train_loss = 0
        for i, (x, y) in enumerate(tqdm(train_loader)):
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

        model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            for i, (x, y) in enumerate(tqdm(val_loader)):
                x = x.to(cfg.device)
                y = y.to(cfg.device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
                _, predicted = y_pred.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

            val_loss /= len(val_loader)
            val_acc = 100 * correct / total

            wandb_logger.log_metrics({"val_loss": val_loss, "val_acc": val_acc})

    model.eval()
    with torch.no_grad():
        test_loss = 0
        correct = 0
        total = 0
        correct_retain = 0
        total_retain = 0
        correct_forget = 0
        total_forget = 0

        for i, (x, y) in enumerate(tqdm(test_loader)):
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            test_loss += loss.item()

            _, predicted = y_pred.max(1)

            # total and correct predictions
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            # filtering retain and forget set
            for idx in range(y.size(0)):
                if y[idx].item() in cfg.forgetting_set:
                    # forget set data
                    total_forget += 1
                    if predicted[idx].item() == y[idx].item():
                        correct_forget += 1
                else:
                    # retain set data
                    total_retain += 1
                    if predicted[idx].item() == y[idx].item():
                        correct_retain += 1

        test_loss /= len(test_loader)
        test_acc = 100 * correct / total
        retain_acc = 100 * correct_retain / total_retain if total_retain > 0 else 0.0
        forget_acc = 100 * correct_forget / total_forget if total_forget > 0 else 0.0

        # results logging
        wandb_logger.log_metrics({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "retain_acc": retain_acc,
            "forget_acc": forget_acc
        })

        if cfg.unlearn.update_json == True:
            with open("src/metrics/metrics.json", "r") as file:
                data = json.load(file)
            done = add_case(data, "original_model", str(cfg.forgetting_set), retain_acc, forget_acc)
            if not done:
                done = update_case(data, "original_model", str(cfg.forgetting_set), retain_acc, forget_acc)
            if not done:
                print("Failed to add/update json")

    #create folder if not exist and save torch model
    os.makedirs(os.path.join(cfg.currentDir, cfg.train.save_path), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg.currentDir, cfg.train.save_path, cfg.dataset.name + '_' + cfg.model + '.pth'))


            
if __name__ == '__main__':
    main()

