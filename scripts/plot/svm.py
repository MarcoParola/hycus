import torch
import hydra
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC  # Usa LinearSVC invece di SVC
import wandb
from omegaconf import DictConfig, OmegaConf
import flatdict


def svm_classifier(X_train, y_train, X_val, y_val, X_test, y_test):
    # Sposta i tensori sulla CPU e converti in NumPy
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

    c_values = [0.1, 1, 10, 100]
    best_c = 1
    best_score = 0

    for c in c_values:
        svm = LinearSVC(C=c)  
        print(f"Training SVM with C={c}")
        svm.fit(X_train, y_train)
        val_score = svm.score(X_val, y_val)
        if val_score > best_score:
            best_score = val_score
            best_c = c

    print(f"Miglior valore di C: {best_c} con accuratezza {best_score:.2f} sulla validazione")

    # Allena il modello con il miglior valore di C
    svm = LinearSVC(C=best_c)
    svm.fit(X_train, y_train)

    # Predizioni sul set di test
    y_pred = svm.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuratezza sul test: {test_accuracy:.2f}")
    print("\nReport di classificazione:")
    print(classification_report(y_test, y_pred))

    # Calcolo delle metriche di classificazione
    class_report = classification_report(y_test, y_pred, output_dict=True)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)

    # Log su wandb
    wandb.log({"Test Accuracy": test_accuracy})

    # Creazione degli istogrammi per Precision, Recall, F1-Score
    classes = [f"Class {i}" for i in range(len(precision))]
    wandb.log({
        "Precision Histogram": wandb.plot.bar(
            wandb.Table(data=[[cls, p] for cls, p in zip(classes, precision)], columns=["Class", "Precision"]),
            "Class",
            "Precision",
            title="Precision per Classe"
        ),
        "Recall Histogram": wandb.plot.bar(
            wandb.Table(data=[[cls, r] for cls, r in zip(classes, recall)], columns=["Class", "Recall"]),
            "Class",
            "Recall",
            title="Recall per Classe"
        ),
        "F1-Score Histogram": wandb.plot.bar(
            wandb.Table(data=[[cls, f] for cls, f in zip(classes, f1)], columns=["Class", "F1-Score"]),
            "Class",
            "F1-Score",
            title="F1-Score per Classe"
        )
    })


@hydra.main(config_path='../config', config_name='config', version_base=None)
def main(cfg):
    # Configurazione di WandB
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    flat_config = dict(flatdict.FlatDict(config_dict, delimiter="/"))
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project+'_svm', config=flat_config)

    # Caricamento dei dati
    if cfg.unlearning_method == 'retrain': #Caso del golden model
        train_features = torch.load('data/features/cifar10/only_retain_set_features'+str(cfg.forgetting_set)+'.pt')
        train_labels = torch.load('data/features/cifar10/only_retain_set_labels'+str(cfg.forgetting_set)+'.pt')
        validation_features = torch.load('data/features/cifar10/only_retain_set_features'+str(cfg.forgetting_set)+'.pt')
        validation_labels = torch.load('data/features/cifar10/only_retain_set_labels'+str(cfg.forgetting_set)+'.pt')
        test_features = torch.load('data/features/cifar10/only_retain_set_features'+str(cfg.forgetting_set)+'.pt')
        test_labels = torch.load('data/features/cifar10/only_retain_set_labels'+str(cfg.forgetting_set)+'.pt')
    else:
        if cfg.original_model==True: #Caso del modello originale
            train_features = torch.load('data/features/cifar10/train_features_original.pt')
            train_labels = torch.load('data/features/cifar10/train_labels_original.pt')
            validation_features = torch.load('data/features/cifar10/val_features_original.pt')
            validation_labels = torch.load('data/features/cifar10/val_labels_original.pt')
            test_features = torch.load('data/features/cifar10/test_features_original.pt')
            test_labels = torch.load('data/features/cifar10/test_labels_original.pt')
        else: #Caso del modello unlearning
            train_features = torch.load('data/features/cifar10/train_features_'+cfg.unlearning_method+'_'+str(cfg.forgetting_set)+'.pt')
            train_labels = torch.load('data/features/cifar10/train_labels_'+cfg.unlearning_method+'_'+str(cfg.forgetting_set)+'.pt')
            validation_features = torch.load('data/features/cifar10/val_features_'+cfg.unlearning_method+'_'+str(cfg.forgetting_set)+'.pt')
            validation_labels = torch.load('data/features/cifar10/val_labels_'+cfg.unlearning_method+'_'+str(cfg.forgetting_set)+'.pt')
            test_features = torch.load('data/features/cifar10/test_features_'+cfg.unlearning_method+'_'+str(cfg.forgetting_set)+'.pt')
            test_labels = torch.load('data/features/cifar10/test_labels_'+cfg.unlearning_method+'_'+str(cfg.forgetting_set)+'.pt')

    # Esecuzione del classificatore SVM
    svm_classifier(train_features, train_labels, validation_features, validation_labels, test_features, test_labels)

    # Finalizza il run su wandb
    wandb.finish()


if __name__ == '__main__':
    main()
