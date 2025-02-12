import torch
import torch.nn.functional as F
import numpy as np
import random
import json
import hydra
from sklearn.svm import SVC
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def compute_predictions(model, loader):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y_true = list()
    y_pred = list()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            _, preds = torch.max(logits, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return y_true, y_pred


def compute_classification_metrics(model, test_loader, num_classes, forgetting_subset):
    
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

def compute_metrics(model, test_loader, num_classes, forgetting_subset):
    classification_metrics = compute_classification_metrics(model, test_loader, num_classes, forgetting_subset)
    print("classification_metrics")
    metrics = {**classification_metrics}
    return metrics


def get_case(data, dataset, unlearning_method, forgetting_set):
    """
    Retrieve a specific case for a specific unlearning method and forgetting set, in the json file.
    """
    if dataset not in data["datasets"]:
        return None

    dataset_data = data["datasets"][dataset]

    if unlearning_method == "original_model":
        for case in dataset_data["original_model"]:
            if str(case["forgetting_set"]) == forgetting_set:
                return case
    else:
        for method in dataset_data["unlearning_methods"]:
            if method["method_name"] == unlearning_method:
                for case in method["cases"]:
                    if str(case["forgetting_set"]) == forgetting_set:
                        return case
    return None

def update_case(data, dataset, unlearning_method, forgetting_set, new_accuracy_retain, new_accuracy_forget):
    """
    Update the accuracy of a specific case for a specific unlearning method and forgetting set, in the json file.
    """
    if dataset not in data["datasets"]:
        return False

    dataset_data = data["datasets"][dataset]

    if unlearning_method == "original_model":
        for caso in dataset_data["original_model"]:
            if caso["forgetting_set"] == forgetting_set:
                caso["accuracy_retain"] = new_accuracy_retain
                with open("src/metrics/metrics.json", "w") as file:
                    json.dump(data, file, indent=2)
                return True
    else:
        for method in dataset_data["unlearning_methods"]:
            if method["method_name"] == unlearning_method:
                for caso in method["cases"]:
                    if str(caso["forgetting_set"]) == forgetting_set:
                        caso["accuracy_retain"] = new_accuracy_retain
                        caso["accuracy_forget"] = new_accuracy_forget
                        with open("src/metrics/metrics.json", "w") as file:
                            json.dump(data, file, indent=2)
                        return True

    return False

def add_case(data, dataset, unlearning_method, forgetting_set, accuracy_retain, accuracy_forget):
    """
    Add a new case for a specific unlearning method and forgetting set, in the json file.
    """
    if dataset not in data["datasets"]:
        return False

    dataset_data = data["datasets"][dataset]

    if unlearning_method == "original_model":
        # check if the case already exists
        for caso in dataset_data["original_model"]:
            if caso["forgetting_set"] == forgetting_set:
                return False  # Caso già esistente
        # update the original model case
        dataset_data["original_model"].append({
            "forgetting_set": forgetting_set,
            "accuracy_retain": accuracy_retain
        })
        with open("src/metrics/metrics.json", "w") as file:
            json.dump(data, file, indent=2)
        return True
    else:
        for method in dataset_data["unlearning_methods"]:
            if method["method_name"] == unlearning_method:
                # check if the case already exists
                for case in method["cases"]:
                    if str(case["forgetting_set"]) == forgetting_set:
                        return False  # case already exists
                # add the new case
                method["cases"].append({
                    "forgetting_set": forgetting_set,
                    "accuracy_retain": accuracy_retain,
                    "accuracy_forget": accuracy_forget
                })
                with open("src/metrics/metrics.json", "w") as file:
                    json.dump(data, file, indent=2)
                return True

    return False

def calculate_l2_distance(model_path_1, model_path_2):
    state_dict_1 = torch.load(model_path_1)
    state_dict_2 = torch.load(model_path_2)

    if state_dict_1.keys() != state_dict_2.keys():
        raise ValueError("The two models have different structures or different keys in their state_dict.")

    l2_distance = 0.0
    for key in state_dict_1.keys():
        weight_diff = state_dict_1[key] - state_dict_2[key]
        l2_distance += torch.sum(weight_diff ** 2).item()
    
    l2_distance = l2_distance ** 0.5
    return l2_distance


def calculate_shannon_divergence(features_path_1, features_path_2):
    features_1 = torch.load(features_path_1)
    features_2 = torch.load(features_path_2)

    if features_1.shape != features_2.shape:
        raise ValueError("The two feature tensors have different shapes.")

    P = F.softmax(features_1, dim=-1)
    Q = F.softmax(features_2, dim=-1)

    kl_pq = torch.sum(P * torch.log(P / Q), dim=-1)
    kl_qp = torch.sum(Q * torch.log(Q / P), dim=-1)
    shannon_divergence = 0.5 * (kl_pq + kl_qp).mean().item()
    return shannon_divergence

def calculate_aus(data, dataset, unlearning_method, forgetting_set):
    if dataset not in data["datasets"]:
        raise ValueError(f"Dataset '{dataset}' not found.")
    dataset_data = data["datasets"][dataset]
    original = get_case(dataset_data, "original_model", forgetting_set)
    unl = get_case(dataset_data, unlearning_method, forgetting_set)
    if not original:
        raise ValueError(f"Original model case not found for forgetting_set {forgetting_set}.")
    if not unl:
        raise ValueError(f"Case not found for the method '{unlearning_method}' and forgetting_set {forgetting_set}.")
    return (1 - (original["accuracy_retain"] - unl["accuracy_retain"])) / (1 + abs(unl["accuracy_forget"]))



@hydra.main(config_path='../../config', config_name='config', version_base=None)
def main(cfg):
    """
    for method in ["icus", "scrub", "ssd", "badT"]:
        aus=calculate_aus(method, str(cfg.forgetting_set))
        print("Method: "+method+" AUS: "+str(aus))
    
    #TEST OF JSON UPDATE

    print("Modifica test")
    with open("src/metrics/metrics.json", "r") as file:
        data = json.load(file)
    done = update_case(data, "ssd", str(cfg.forgetting_set), 0.6, 0.1)
    print("Modifica test 2")
    done = add_case(data, "ssd", "[2, 3, 4]", 0.01, 0.1)
    print(done)
    """

    #Here for shannon divergence and l2 distance
    if cfg.dataset.name == "cifar100":
        unlearning_method = ["icus", "scrub", "badT", "icus_hierarchy", "finetuning"]
    elif cfg.dataset.name == "cifar10" and int(cfg.forgetting_set_size) > 2:
        unlearning_method = ["icus", "scrub", "badT","finetuning"]
    elif cfg.dataset.name == "cifar10" and int(cfg.forgetting_set_size) <= 2:
        unlearning_method = ["icus", "scrub", "ssd", "badT", "finetuning"]
    elif cfg.dataset.name == "lfw":
        unlearning_method = ["icus", "scrub", "badT"]
    else:
        raise ValueError("Not supported settings")
    
    for u in unlearning_method:
        l2_distance = calculate_l2_distance("checkpoints/"+cfg.dataset.name+"_"+cfg.model+"_only_retain_set"+str(cfg.forgetting_set)+".pth", "checkpoints/"+cfg.dataset.name+"_forgetting_set_"+str(cfg.forgetting_set)+"_"+u+"_"+cfg.model+".pth")
        print("L2 distance "+u+": ", l2_distance)
        shannon_divergence = calculate_shannon_divergence("data/features/"+cfg.dataset.name+"/test_features_"+u+"_"+str(cfg.forgetting_set)+".pt", "data/features/"+cfg.dataset.name+"/test_features_only_retain_forgetting_"+str(cfg.forgetting_set)+".pt")
        print("Shannon divergence "+u+": ", shannon_divergence)

if __name__ == '__main__':
    main()
