import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import wandb
import flatdict
from omegaconf import OmegaConf
import hydra
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import os
import numpy as np

from src.datasets.dataset import load_dataset
from src.models.model import load_model

def knn(X_train, y_train, X_val, y_val, X_test, y_test, cfg):

    # Standardizzazione dei dati
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Seleziona il miglior valore di K
    k_values = list(range(5, 6))
    best_k = 11

    # Allena il modello con il miglior valore di k
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)

    # Predizioni sul set di test
    y_pred = knn.predict(X_test)
    
    return y_pred



def svm_classifier(X_train, y_train, X_val, y_val, X_test, y_test):

    # Standardizzazione dei dati
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

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

    # Creazione dei grafici a barre
    precision_table = wandb.Table(data=[[str(i), p] for i, p in enumerate(precision)], columns=["Class", "Precision"])
    recall_table = wandb.Table(data=[[str(i), r] for i, r in enumerate(recall)], columns=["Class", "Recall"])
    f1_table = wandb.Table(data=[[str(i), f] for i, f in enumerate(f1)], columns=["Class", "F1-Score"])


@hydra.main(config_path='../../config', config_name='config', version_base=None)
def main(cfg):
    
    train_features_345 = torch.load('data/features/fs[3,4,5]/icus/train_features_icus_[3, 4, 5].pt').cpu().numpy()
    train_labels_345 = torch.load('data/features/fs[3,4,5]/icus/train_labels_icus_[3, 4, 5].pt').cpu().numpy()
    test_features_345 = torch.load('data/features/fs[3,4,5]/icus/test_features_icus_[3, 4, 5].pt').cpu().numpy()
    test_labels_345 = torch.load('data/features/fs[3,4,5]/icus/test_labels_icus_[3, 4, 5].pt').cpu().numpy()
    train_features_orig = torch.load('data/features/original/train_features_original.pt').cpu().numpy()
    train_labels_orig = torch.load('data/features/original/train_labels_original.pt').cpu().numpy()
    test_features_orig = torch.load('data/features/original/test_features_original.pt').cpu().numpy()
    test_labels_orig = torch.load('data/features/original/test_labels_original.pt').cpu().numpy()


    # Esegui il classificatore KNN con logging su wandb
    predictions_knn_345 = knn(train_features_345, train_labels_345, test_features_345, test_labels_345, test_features_345, test_labels_345, cfg)
    #predictions_svm_345 = svm_classifier(train_features_345, train_labels_345, test_features_345, test_labels_345, test_features_345, test_labels_345)

    predictions_knn_orig = knn(train_features_orig, train_labels_orig, test_features_orig, test_labels_orig, test_features_orig, test_labels_orig, cfg)
    #predictions_svm_orig = svm_classifier(train_features_orig, train_labels_orig, test_features_orig, test_labels_orig, test_features_orig, test_labels_orig)

    print(predictions_knn_345, predictions_knn_orig)

    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    train, val, test = load_dataset('cifar10', data_dir, cfg.dataset.resize)
    print(len(train), len(val), len(test))

    model_folder = 'checkpoints'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_orig = load_model('ResNet18_Weights.IMAGENET1K_V1', f'{model_folder}/cifar10_resnet.pth').to(device)
    model_345 = load_model('ResNet18_Weights.IMAGENET1K_V1', f'{model_folder}/cifar10_resnet_only_retain_set[3, 4, 5].pth').to(device)

    pred_orig,pred_345,test_labels_orig = [],[],[]

    model_orig.eval()
    model_345.eval()

    with torch.no_grad():
        for j, (x, y) in enumerate(test):
            print(j)
            x = x.unsqueeze(0).to(device)
            y = torch.tensor(y).to(device)
            out_orig = model_orig(x)
            out_345 = model_345(x)
            pred_orig.append(out_orig.cpu().argmax(dim=1))
            pred_345.append(out_345.cpu().argmax(dim=1))
            test_labels_orig.append(y.cpu())
            torch.cuda.empty_cache()

    pred_orig = torch.cat(pred_orig).numpy()
    pred_345 = torch.cat(pred_345).numpy()
    print(pred_orig.shape, pred_345.shape)
    print(test_labels_orig)
    test_labels_orig = torch.tensor(test_labels_orig).numpy()
    print(pred_orig.shape, pred_345.shape, test_labels_orig.shape)
    predictions_knn_orig = torch.tensor(predictions_knn_orig).numpy()
    predictions_knn_345 = torch.tensor(predictions_knn_345).numpy()
    
    # draw and save confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm_orig = confusion_matrix(test_labels_orig, pred_orig)
    cm_345 = confusion_matrix(test_labels_orig, pred_345)
    cm_knn_orig = confusion_matrix(test_labels_orig, predictions_knn_orig)
    cm_knn_345 = confusion_matrix(test_labels_orig, predictions_knn_345)
    # cm_svm_orig = confusion_matrix(test_labels_orig, predictions_svm_orig)
    # cm_svm_345 = confusion_matrix(test_labels_orig, predictions_svm_345)

    # scale cm
    cm_orig = cm_orig.astype('float') / cm_orig.sum(axis=1)[:, np.newaxis]
    cm_345 = cm_345.astype('float') / cm_345.sum(axis=1)[:, np.newaxis]
    cm_knn_orig = cm_knn_orig.astype('float') / cm_knn_orig.sum(axis=1)[:, np.newaxis]
    cm_knn_345 = cm_knn_345.astype('float') / cm_knn_345.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    # draw confusion matrix using 0.xx format and hide color bar
    sns.heatmap(cm_orig, annot=True, ax=ax[0, 0], fmt='.2f', cmap='Blues', cbar=False)
    ax[0, 0].set_title('Original Features')
    sns.heatmap(cm_345, annot=True, ax=ax[0, 1], fmt='.2f', cmap='Blues', cbar=False)
    ax[0, 1].set_title('Retained Features')
    sns.heatmap(cm_knn_orig, annot=True, ax=ax[1, 0], fmt='.2f', cmap='Blues', cbar=False)
    ax[1, 0].set_title('KNN with Original Features')
    sns.heatmap(cm_knn_345, annot=True, ax=ax[1, 1], fmt='.2f', cmap='Blues', cbar=False)
    ax[1, 1].set_title('KNN with Retained Features')

    # sns.heatmap(cm_svm_orig, annot=True, ax=ax[1, 1])
    # ax[1, 1].set_title('SVM with Original Features')
    # sns.heatmap(cm_svm_345, annot=True, ax=ax[1, 2])
    # ax[1, 2].set_title('SVM with Retained Features')


    plt.show()

if __name__ == '__main__':
    main()

    



