import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import hydra
import torchvision
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.datasets.dataset import load_dataset
from src.models.classifier import Classifier

# Funzione per calcolare la matrice di confusione
def compute_confusion_matrix(model, data_loader, cfg, unlearned=False, device='cpu'):
    model.eval()  # Modalità di valutazione
    device = cfg.device
    model.to(device)

    # Liste per raccogliere le etichette vere e le predizioni
    y_true = []
    y_pred = []

    # Softmax
    softmax = torch.nn.Softmax(dim=1)  # Creazione della funzione Softmax lungo la dimensione delle classi

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
    disp.plot(cmap=plt.cm.Blues, values_format='d')  # Usa una mappa colori e formato intero
    plt.title("Confusion Matrix")

    # Salva la matrice di confusione in base al fatto che si stia facendo unlearning
    if unlearned:
        plt.savefig(f"src/data/confusion_matrix_postUnl_{cfg.unlearning_method}.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"src/data/confusion_matrix_preUnl_{cfg.unlearning_method}.png", dpi=300, bbox_inches='tight')
    
    plt.close()  # Chiudi la figura per evitare sovrapposizioni
    return cm


@hydra.main(config_path='../config', config_name='config', version_base=None)
def main(cfg):
    # Percorso dei dati
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((224, 224)),  # Ridimensionamento delle immagini
    ])

    # Carica il dataset CIFAR-10
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    # Crea il DataLoader per il test
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

    # Carica il modello
    model = Classifier(cfg.weights_name, num_classes=cfg[cfg.dataset.name].n_classes, finetune=True)
    model.to(cfg.device)

    # Carica i pesi del modello pre-unlearning
    weights = os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_{cfg.model}.pth")
    model.load_state_dict(torch.load(weights, map_location=cfg.device))

    # Calcola e salva la matrice di confusione pre-unlearning
    compute_confusion_matrix(model, test_loader, cfg)

    # Carica i pesi del modello post-unlearning
    unlearned_weights = os.path.join(cfg.currentDir, cfg.train.save_path, f"{cfg.dataset.name}_forgetting_size_{cfg.forgetting_set_size}_{cfg.unlearning_method}_{cfg.model}.pth")
    model.load_state_dict(torch.load(unlearned_weights, map_location=cfg.device))

    # Calcola e salva la matrice di confusione post-unlearning
    compute_confusion_matrix(model, test_loader, cfg, unlearned=True)


if __name__ == '__main__':
    main()
