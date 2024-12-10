import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import wandb
import flatdict
from omegaconf import OmegaConf
import hydra
from sklearn.neighbors import KNeighborsClassifier

def knn(X_train, y_train, X_val, y_val, X_test, y_test, cfg):
    # Converti i tensori in numpy array e spostali sulla CPU
    X_train = X_train.cpu().numpy()
    y_train = y_train.cpu().numpy()
    X_val = X_val.cpu().numpy()
    y_val = y_val.cpu().numpy()
    X_test = X_test.cpu().numpy()
    y_test = y_test.cpu().numpy()

    # Standardizzazione dei dati
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Seleziona il miglior valore di K
    k_values = list(range(5, 6))
    best_k = 1
    best_score = 0

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        val_score = knn.score(X_val, y_val)
        if val_score > best_score:
            best_score = val_score
            best_k = k

    print(f"Miglior valore di K: {best_k} con accuratezza {best_score:.2f}")

    # Allena il modello con il miglior valore di k
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)

    # Predizioni sul set di test
    y_pred = knn.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuratezza sul test: {test_accuracy:.2f}")

    # Calcolo delle metriche di classificazione
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)

    print("\nReport di classificazione:")
    print(classification_report(y_test, y_pred))

    # Log su wandb
    wandb.log({"Test Accuracy": test_accuracy})

    # Creazione dei grafici a barre
    precision_table = wandb.Table(data=[[str(i), p] for i, p in enumerate(precision)], columns=["Class", "Precision"])
    recall_table = wandb.Table(data=[[str(i), r] for i, r in enumerate(recall)], columns=["Class", "Recall"])
    f1_table = wandb.Table(data=[[str(i), f] for i, f in enumerate(f1)], columns=["Class", "F1-Score"])

    # Log su wandb come grafico a barre
    wandb.log({
        "Precision by Class": wandb.plot.bar(
            precision_table, "Class", "Precision", title="Precision per Classe"
        ),
        "Recall by Class": wandb.plot.bar(
            recall_table, "Class", "Recall", title="Recall per Classe"
        ),
        "F1-Score by Class": wandb.plot.bar(
            f1_table, "Class", "F1-Score", title="F1-Score per Classe"
        )
    })

@hydra.main(config_path='../config', config_name='config', version_base=None)
def main(cfg):
    # Converte la configurazione in un dizionario
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    # Appiattisce la configurazione per wandb (ad esempio, nested dict -> single level dict)
    flat_config = dict(flatdict.FlatDict(config_dict, delimiter="/"))
    
    # Inizializza wandb e registra i parametri
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project+'_knn', config=flat_config)
    
    # Caricamento dei dati
    if cfg.golden_model==True:
        train_features = torch.load('data/features/cifar10/train_features_only_retain_forgetting_'+str(cfg.forgetting_set)+'.pt')
        train_labels = torch.load('data/features/cifar10/train_labels_only_retain_forgetting_'+str(cfg.forgetting_set)+'.pt')
        validation_features = torch.load('data/features/cifar10/val_features_only_retain_forgetting_'+str(cfg.forgetting_set)+'.pt')
        validation_labels = torch.load('data/features/cifar10/val_labels_only_retain_forgetting_'+str(cfg.forgetting_set)+'.pt')
        test_features = torch.load('data/features/cifar10/test_features_only_retain_forgetting_'+str(cfg.forgetting_set)+'.pt')
        test_labels = torch.load('data/features/cifar10/test_labels_only_retain_forgetting_'+str(cfg.forgetting_set)+'.pt')
    else:
        if cfg.orignal_model==True:
        train_features = torch.load('data/features/cifar10/train_features_original.pt')
        train_labels = torch.load('data/features/cifar10/train_labels_original.pt')
        validation_features = torch.load('data/features/cifar10/val_features_original.pt')
        validation_labels = torch.load('data/features/cifar10/val_labels_original.pt')
        test_features = torch.load('data/features/cifar10/test_features_original.pt')
        test_labels = torch.load('data/features/cifar10/test_labels_original.pt')
        else:
            train_features = torch.load('data/features/cifar10/train_features_'+cfg.unlearning_method+'_'+str(cfg.forgetting_set)+'.pt')
            train_labels = torch.load('data/features/cifar10/train_labels_'+cfg.unlearning_method+'_'+str(cfg.forgetting_set)+'.pt')
            validation_features = torch.load('data/features/cifar10/val_features_'+cfg.unlearning_method+'_'+str(cfg.forgetting_set)+'.pt')
            validation_labels = torch.load('data/features/cifar10/val_labels_'+cfg.unlearning_method+'_'+str(cfg.forgetting_set)+'.pt')
            test_features = torch.load('data/features/cifar10/test_features_'+cfg.unlearning_method+'_'+str(cfg.forgetting_set)+'.pt')
            test_labels = torch.load('data/features/cifar10/test_labels_'+cfg.unlearning_method+'_'+str(cfg.forgetting_set)+'.pt')
        

    # Esegui il classificatore KNN con logging su wandb
    knn(train_features, train_labels, validation_features, validation_labels, test_features, test_labels, cfg)

    # Finalizza il run su wandb
    wandb.finish()

if __name__ == '__main__':
    main()


