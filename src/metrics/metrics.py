import torch
import torch.nn.functional as F
import numpy as np
import random
from sklearn.svm import SVC
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



def compute_predictions(model, loader, test):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y_true = list()
    y_pred = list()
    with torch.no_grad():
        if test:
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                _, preds = torch.max(logits, 1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        else:
            for x, y, _ in loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                _, preds = torch.max(logits, 1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
    return y_true, y_pred


def compute_classification_metrics(model, test_loader, num_classes, forgetting_subset, test):
    print("Starting metrics computation")
    y_true, y_pred = compute_predictions(model, test_loader, test)

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

def compute_metrics(model, test_loader, num_classes, forgetting_subset, test = True):
    classification_metrics = compute_classification_metrics(model, test_loader, num_classes, forgetting_subset, test)
    mia_metrics = None # TODO implement MIA metrics
    #metrics = {**classification_metrics, **mia_metrics}
    print("classification_metrics")
    metrics = {**classification_metrics}
    return metrics


def prepare_membership_inference_attack(test_loader,forgetting_subset, batch_size):
    forget_indices_test = [i for i in range(len(test_loader)) if test_loader[i][1] in forgetting_subset]
    forget_indices_test_loader = DataLoader(test_loader, batch_size=batch_size, sampler=SubsetRandomSampler(forget_indices_test))
    return forget_indices_test_loader

##### CODE FROM https://github.com/ayushkumartarun/zero-shot-unlearning/blob/main/metrics.py ###### 
def entropy(p, dim = -1, keepdim = False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

def collect_prob(data_loader, model):   
    data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=1, shuffle=False, num_workers = 8, prefetch_factor = 10)
    prob = []
    with torch.no_grad():
        for batch in data_loader:
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            data, target = batch
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)


def compute_mia(test_forget_loader, train_forget_loader, model, num_classes, loggers):
    attack_result = get_membership_attack_prob(test_forget_loader, train_forget_loader, model)
    loggers.log_metrics({"Membership Inference Attack": attack_result}, step=0)
    print("Membership Inference Attack: ", attack_result)



def get_membership_attack_prob(test_loader, train_loader, model):
    X_ts, Y_ts, X_tr, Y_tr = get_membership_attack_data(test_loader, train_loader, model)
    clf = SVC(C=3,gamma='auto',kernel='rbf')
    clf.fit(X_tr, Y_tr)

    size_test=len(X_ts)
    size_train=len(X_tr)
    prob_test = size_test / (size_test + size_train)
    results = []
    for i in range(size_test):
        b = np.random.rand() 
        if b < prob_test:
            j = random.randint(0, size_test-1)
            sample = X_ts[j]
        else:
            j = random.randint(0, size_train-1)
            sample = X_tr[j]
        pred = clf.predict([sample])
        results.append(pred[0])
    return np.mean(results)


def get_membership_attack_data(test_loader, train_loader, model):
    test_prob = collect_prob(test_loader, model)
    train_prob = collect_prob(train_loader, model)
    X_ts = entropy(test_prob).cpu().numpy().reshape(-1, 1)
    X_tr = entropy(train_prob).cpu().numpy().reshape(-1, 1)
    Y_ts = np.concatenate([np.zeros(len(test_prob))])
    Y_tr = np.concatenate([np.ones(len(train_prob))])
    return X_ts, Y_ts, X_tr, Y_tr




#Following functions currently not used
"""
def compute_mia(retain_loader, forget_loader, test_loader, model, num_classes, forgetting_subset, loggers, current_step):
    attack_result = get_membership_attack_prob(retain_loader, forget_loader, test_loader, model)
    metrics_test = compute_metrics(model, test_loader, num_classes, forgetting_subset)
    accuracy_test = metrics_test['accuracy']
    metrics = {
        "accuracy_test": accuracy_test,
        "membership_inference_attack_result": attack_result
    }
    print("Model metrics post-unlearning on test set and membership inference attack:")
    loggers.log_metrics({"Membership Inference Attack": metrics["membership_inference_attack_result"]}, step=0)
    print(metrics)


def get_membership_attack_data(retain_loader, forget_loader, test_loader, model):    
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)
    
    X_r = torch.cat([entropy(retain_prob), entropy(test_prob)]).cpu().numpy().reshape(-1, 1)
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])
    
    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])    
    return X_f, Y_f, X_r, Y_r

def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model):
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(retain_loader, forget_loader, test_loader, model)
    clf = SVC(C=3,gamma='auto',kernel='rbf')
    #clf = LogisticRegression(class_weight='balanced',solver='lbfgs',multi_class='multinomial')
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    return results.mean()

def relearn_time(model, train_loader, valid_loader, reqAcc, lr):
    # measuring relearn time for gold standard model
    rltime = 0
    curr_Acc = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    
    # we will try the relearning step till 4 epochs.
    for epoch in range(10):
        
        for batch in train_loader:
            model.train()
            loss = training_step(model, batch)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            history = [evaluate(model, valid_dl)]
            curr_Acc = history[0]["Acc"]*100
            print(curr_Acc, sep=',')        
            
            rltime += 1
            if(curr_Acc >= reqAcc):
                break
                
        if(curr_Acc >= reqAcc):
            break
    return rltime

def ain(full_model, model, gold_model, train_data, val_retain, val_forget, 
                  batch_size = 256, error_range = 0.05, lr = 0.001):
    # measuring performance of fully trained model on forget class
    forget_valid_dl = DataLoader(val_forget, batch_size)
    history = [evaluate(full_model, forget_valid_dl)]
    AccForget = history[0]["Acc"]*100
    
    print("Accuracy of fully trained model on forget set is: {}".format(AccForget))
    
    retain_valid_dl = DataLoader(val_retain, batch_size)
    history = [evaluate(full_model, retain_valid_dl)]
    AccRetain = history[0]["Acc"]*100
    
    print("Accuracy of fully trained model on retain set is: {}".format(AccRetain))
    
    history = [evaluate(model, forget_valid_dl)]
    AccForget_Fmodel = history[0]["Acc"]*100
    
    print("Accuracy of forget model on forget set is: {}".format(AccForget_Fmodel))
    
    history = [evaluate(model, retain_valid_dl)]
    AccRetain_Fmodel = history[0]["Acc"]*100
    
    print("Accuracy of forget model on retain set is: {}".format(AccRetain_Fmodel))
    
    history = [evaluate(gold_model, forget_valid_dl)]
    AccForget_Gmodel = history[0]["Acc"]*100
    
    print("Accuracy of gold model on forget set is: {}".format(AccForget_Gmodel))
    
    history = [evaluate(gold_model, retain_valid_dl)]
    AccRetain_Gmodel = history[0]["Acc"]*100
    
    print("Accuracy of gold model on retain set is: {}".format(AccRetain_Gmodel))
    
    reqAccF = (1-error_range)*AccForget
    
    print("Desired Accuracy for retrain time with error range {} is {}".format(error_range, reqAccF))
    
    train_loader = DataLoader(train_ds, batch_size, shuffle = True)
    valid_loader = DataLoader(val_forget, batch_size)
    rltime_gold = relearn_time(model = gold_model, train_loader = train_loader, valid_loader = valid_loader, 
                               reqAcc = reqAccF,  lr = lr)
    
    print("Relearning time for Gold Standard Model is {}".format(rltime_gold))
    
    rltime_forget = relearn_time(model = model, train_loader = train_loader, valid_loader = valid_loader, 
                               reqAcc = reqAccF, lr = lr)
    
    print("Relearning time for Forget Model is {}".format(rltime_forget))
    
    rl_coeff = rltime_forget/rltime_gold
    print("AIN = {}".format(rl_coeff))"""