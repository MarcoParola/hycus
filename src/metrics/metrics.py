import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_predictions(model, test_loader):
    """Compute classification metrics

    model: torch.nn.Module: model
    test_loader: torch.utils.data.DataLoader: test data loader
    """
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y_true = list()
    y_pred = list()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            _, preds = torch.max(logits, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return y_true, y_pred


def compute_classification_metrics(model, test_loader, num_classes, forgetting_subset):
    """Compute classification metrics

    model: torch.nn.Module: model
    test_loader: torch.utils.data.DataLoader: test data loader
    num_classes: int: number of classes
    forgetting_subset: list: class subset
    """
    y_true, y_pred = compute_predictions(model, test_loader)

    # compute metrics three times (on whole dataset, on the forgetting subset, on the retaining subset)
    metrics = dict()
    # whole dataset
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    # forgetting subset
    y_true_forgetting = [y_true[i] for i in range(len(y_true)) if y_true[i] in forgetting_subset]
    y_pred_forgetting = [y_pred[i] for i in range(len(y_pred)) if y_true[i] in forgetting_subset]
    metrics['accuracy_forgetting'] = accuracy_score(y_true_forgetting, y_pred_forgetting)
    # retaining subset
    y_true_retaining = [y_true[i] for i in range(len(y_true)) if y_true[i] not in forgetting_subset]
    y_pred_retaining = [y_pred[i] for i in range(len(y_pred)) if y_true[i] not in forgetting_subset]
    metrics['accuracy_retaining'] = accuracy_score(y_true_retaining, y_pred_retaining)
    
    return metrics



