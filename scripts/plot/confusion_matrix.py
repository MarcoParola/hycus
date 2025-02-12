import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import hydra
import torchvision
from datetime import datetime
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.datasets.dataset import load_dataset
from src.models.classifier import Classifier
from scripts.descr_and_similarity import calculate_embeddings, calculate_dissimilarity
from matplotlib.colors import TwoSlopeNorm

# Function to compute the confusion matrix
def compute_confusion_matrix(model, data_loader, cfg, save_plot=True, unlearned=False, device='cpu'):
    model.eval()  # Modalità di valutazione
    device = cfg.device
    model.to(device)

    y_true = []
    y_pred = []

    # Softmax
    softmax = torch.nn.Softmax(dim=1)  # Creazione della funzione Softmax lungo la dimensione delle classi
    if cfg.dataset.name == 'cifar100':
        plt.rcParams["figure.figsize"] = (30, 30)

    with torch.no_grad():  # Disabilita il calcolo del gradiente durante la valutazione
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)  # Output del modello (logits, non probabilità)
            
            # Applicare Softmax per ottenere le probabilità
            probs = softmax(logits)
            
            # Le predizioni sono la classe con la probabilità massima
            preds = torch.argmax(probs, dim=1)

            # Aggiungi i risultati alle liste
            y_true.extend(y.cpu().numpy())  # Etichette vere
            y_pred.extend(preds.cpu().numpy())  # Predizioni

    # Calcola la matrice di confusione
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(0, cfg.dataset.classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, values_format="", include_values=False)
    
    if unlearned:
        plt.title("Confusion Matrix "+cfg.unlearning_method +" forgetting size "+str(cfg.forgetting_set))
    else:
        plt.title("Confusion Matrix pre-unlearning")

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Salva la matrice di confusione in base al fatto che si stia facendo unlearning
    if save_plot:
        if unlearned:
            plt.savefig(f"src/data/confusion_matrix_postUnl_{cfg.unlearning_method}_{str(cfg.forgetting_set)}.png", dpi=300, bbox_inches='tight')
        else:
            if cfg.golden_model==True:
                plt.savefig(f"src/data/confusion_matrix_golden_{str(cfg.forgetting_set)}.png", dpi=300, bbox_inches='tight')
            else:
                plt.savefig(f"src/data/confusion_matrix_preUnl_{cfg.unlearning_method}_{str(cfg.forgetting_set)}_{cfg.unlearning_method}.png", dpi=300, bbox_inches='tight')
    
    plt.close()  # Chiudi la figura per evitare sovrapposizioni
    return cm


def difference_between_matrices(cm1, cm2):
    if cm1.shape != cm2.shape:
        raise ValueError("Le due confusion matrix devono avere le stesse dimensioni.")
    # Calcola la differenza
    cm_diff = cm2 - cm1
    return cm_diff


def plot_multiple_confusion_matrices(cms, cfg, names, labels=None, rows=2, cols=5, positions=None):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()  # Converti la griglia di assi in un array 1D

    # Se le posizioni non sono specificate, usa un ordine sequenziale
    if positions is None:
        positions = list(range(len(cms)))

    # Trova i valori massimo e minimo tra tutte le confusion matrix per mantenere la scala coerente
    overall_min = min(cm.min() for cm in cms)
    overall_max = max(cm.max() for cm in cms)

    # Crea una normalizzazione centrata su 0
    norm = TwoSlopeNorm(vmin=overall_min, vcenter=0, vmax=overall_max)

    # Aggiungi confusion matrix nelle posizioni specificate
    for i, pos in enumerate(positions):
        if i < len(cms) and pos < len(axes):
            ax = axes[pos]
            # Usa la mappa di colori con TwoSlopeNorm
            im = ax.imshow(cms[i], interpolation='nearest', cmap=plt.cm.bwr, norm=norm)
            ax.set_title(f"Matrix {names[i]}")
            if labels is not None:
                ax.set_xticks(np.arange(len(labels)))
                ax.set_yticks(np.arange(len(labels)))
                ax.set_xticklabels(labels)
                ax.set_yticklabels(labels)

            # Etichette
            if cfg.dataset.name == 'cifar10':
                for j in range(cms[i].shape[0]):
                    for k in range(cms[i].shape[1]):
                        ax.text(k, j, f"{cms[i][j, k]}", ha="center", va="center", color="black")

    # Nascondi gli assi non utilizzati
    for j in range(len(cms), len(axes)):
        axes[j].axis('off')

    # Salva la figura
    plt.savefig("src/data/differences_between_confusion_matrix_" + str(cfg.forgetting_set) + ".png", dpi=300, bbox_inches='tight')

def calculate_cm_error(test_dataloader, cm, nclasses):
    class_counts = np.zeros(nclasses)  
    for _, labels in test_dataloader:
        for label in labels:
            class_counts[label] += 1
    cm = cm.astype(float)
    for i in range(nclasses):
        cm[i] = cm[i] / class_counts[i]
    error = 0
    for i in range(nclasses):
        for j in range(nclasses):
            error += abs(cm[i][j])
    error = error / (nclasses*nclasses)
    return error

def calculate_weighted_cm_error(test_dataloader, cm, embeddings_dissimilarity, nclasses):
    class_counts = np.zeros(nclasses)  
    for _, labels in test_dataloader:
        for label in labels:
            class_counts[label] += 1
    cm = cm.astype(float)
    for i in range(nclasses):
        cm[i] = cm[i] / class_counts[i]
    error = 0
    for i in range(nclasses):
        for j in range(nclasses):
            error += embeddings_dissimilarity[i][j] * abs(cm[i][j])
    error = error / (nclasses*nclasses)
    return error


@hydra.main(config_path='../../config', config_name='config', version_base=None)
def main(cfg):
    # Percorso dei dati
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((224, 224)),  # Ridimensionamento delle immagini
    ])

    # Carica il dataset CIFAR-10
    if cfg.dataset.name == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    elif cfg.dataset.name == 'cifar100':
        test_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    elif cfg.dataset.name == 'lfw':
        _, _, test_dataset = load_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)

    # Crea il DataLoader per il test
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

    # Carica il modello
    model = Classifier(cfg.weights_name, num_classes=cfg.dataset.classes, finetune=True)
    model.to(cfg.device)

    #ESEGUI PER SINGOLA CONFUSION MATRIX
    
    """# Carica i pesi del modello pre-unlearning
    weights = os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_{cfg.model}.pth")
    model.load_state_dict(torch.load(weights, map_location=cfg.device))

    # Calcola e salva la matrice di confusione pre-unlearning
    compute_confusion_matrix(model, test_loader, cfg)
    
    # Carica i pesi del modello post-unlearning
    unlearned_weights = os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_forgetting_set_{cfg.forgetting_set}_{cfg.unlearning_method}_{cfg.model}.pth")
    model.load_state_dict(torch.load(unlearned_weights, map_location=cfg.device))

    # Calcola e salva la matrice di confusione post-unlearning
    compute_confusion_matrix(model, test_loader, cfg, unlearned=True)
    """
    
    #ESEGUI PER DIFFERENZA TRA LE CONFUSION MATRIX
    cms=[]
    cm1=cm2=cm3=cm4=cm5=cm6=cm11=cm12=None
    names=[]
    #original
    weights= os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_{cfg.model}.pth")
    if os.path.exists(weights):
        names.append("original")
        model.load_state_dict(torch.load(weights, map_location=cfg.device))
        cm1 = compute_confusion_matrix(model, test_loader, cfg, save_plot=False)
        cms.append(cm1)
    
    #scrub
    weights=os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_forgetting_set_{cfg.forgetting_set}_scrub_{cfg.model}.pth")
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location=cfg.device))
        cm2 = compute_confusion_matrix(model, test_loader, cfg, save_plot=False)
        cms.append(cm2)
        names.append("scrub")

    #ssd
    if cfg.forgetting_set_size < 3:
        weights=os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_forgetting_set_{cfg.forgetting_set}_ssd_{cfg.model}.pth")
        if os.path.exists(weights):
            model.load_state_dict(torch.load(weights, map_location=cfg.device))
            cm3 = compute_confusion_matrix(model, test_loader, cfg, save_plot=False)
            cms.append(cm3)
            names.append("ssd")

    #badT
    weights=os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_forgetting_set_{cfg.forgetting_set}_badT_{cfg.model}.pth")
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location=cfg.device))
        cm4 = compute_confusion_matrix(model, test_loader, cfg, save_plot=False)
        cms.append(cm4)
        names.append("badT")

    #icus
    weights=os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_forgetting_set_{cfg.forgetting_set}_icus_{cfg.model}.pth")
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location=cfg.device))
        cm5 = compute_confusion_matrix(model, test_loader, cfg, save_plot=False)
        cms.append(cm5)
        names.append("icus")
    
    #icus hierarchy
    weights=os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_forgetting_set_{cfg.forgetting_set}_icus_hierarchy_{cfg.model}.pth")
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location=cfg.device))
        cm11 = compute_confusion_matrix(model, test_loader, cfg, save_plot=False)
        cms.append(cm11)
        names.append("icus hierarchy")

    #finetuning
    weights=os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_forgetting_set_{cfg.forgetting_set}_finetuning_{cfg.model}.pth")
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location=cfg.device))
        cm12 = compute_confusion_matrix(model, test_loader, cfg, save_plot=False)
        cms.append(cm12)
        names.append("finetuning")

    #golden
    weights=os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_{cfg.model}_only_retain_set{cfg.forgetting_set}.pth")
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location=cfg.device))
        cm6 = compute_confusion_matrix(model, test_loader, cfg, save_plot=False)
        cms.append(cm6)
        names.append("golden")

    embeddings = calculate_embeddings(cfg.dataset.name)
    embeddings = embeddings.mean(dim=1)
    embeddings_dissimilarity = calculate_dissimilarity(embeddings)

    #differenza tra golden e scrub
    if cm2 is not None:
        cm7 = difference_between_matrices(cm6, cm2)
        cms.append(cm7)
        cm_aux=cm7
        print("Unweighted error scrub: ", calculate_cm_error(test_loader, cm_aux, cfg.dataset.classes))
        print("Weighted scrub: ", calculate_weighted_cm_error(test_loader, cm_aux, embeddings_dissimilarity, cfg.dataset.classes))
        names.append("golden-scrub")

    #differenza tra golden e ssd
    if cm3 is not None:
        cm8 = difference_between_matrices(cm6, cm3)
        cms.append(cm8)
        cm_aux=cm8
        print("Unweighted error ssd: ", calculate_cm_error(test_loader, cm_aux, cfg.dataset.classes))
        print("Weighted error ssd: ", calculate_weighted_cm_error(test_loader, cm_aux, embeddings_dissimilarity, cfg.dataset.classes))
        names.append("golden-ssd")

    #differenza tra golden e badT
    if cm4 is not None:
        cm9 = difference_between_matrices(cm6, cm4)
        cms.append(cm9)
        cm_aux=cm9
        print("Unweighted error badT: ", calculate_cm_error(test_loader, cm_aux, cfg.dataset.classes))
        print("Weighted error badT: ", calculate_weighted_cm_error(test_loader, cm_aux, embeddings_dissimilarity, cfg.dataset.classes))
        names.append("golden-badT")

    #differenza tra golden e icus
    if os.path.exists(weights):
        cm10 = difference_between_matrices(cm6, cm5)
        cms.append(cm10)
        cm_aux=cm10
        print("Unweighted error icus: ", calculate_cm_error(test_loader, cm_aux, cfg.dataset.classes))
        print("Weighted error icus: ", calculate_weighted_cm_error(test_loader, cm_aux, embeddings_dissimilarity, cfg.dataset.classes))
        names.append("golden-icus")
    
    #differenza tra golden e icus hierarchy
    if cm11 is not None:
        cm12 = difference_between_matrices(cm6, cm11)
        cms.append(cm12)
        cm_aux=cm12
        print("Unweighted error Icus hierarchy: ", calculate_cm_error(test_loader, cm_aux, cfg.dataset.classes))
        print("Weighted error Icus hierarchy: ", calculate_weighted_cm_error(test_loader, cm_aux, embeddings_dissimilarity, cfg.dataset.classes))
        names.append("golden-icus hierarchy")
    
    #differenza tra golden e finetuning
    if cm12 is not None:
        cm13 = difference_between_matrices(cm6, cm12)
        cms.append(cm13)
        cm_aux=cm13
        print("Unweighted error finetuning: ", calculate_cm_error(test_loader, cm_aux, cfg.dataset.classes))
        print("Weighted error finetuning: ", calculate_weighted_cm_error(test_loader, cm_aux, embeddings_dissimilarity, cfg.dataset.classes))
        names.append("golden-finetuning")

    plot_multiple_confusion_matrices(cms, cfg, names, labels=np.arange(cfg.dataset.classes), rows=2, cols=len(names)//2)
    

if __name__ == '__main__':
    main()
