import requests
from transformers import BertModel, BertTokenizer
import torch
from descr_and_similarity import calculate_embeddings, calculate_dissimilarity
import matplotlib.pyplot as plt
import numpy as np  
from sklearn.metrics import pairwise_distances

def flatten_class_embeddings(embeddings):
    # Appiattisce la dimensione degli esempi (512) per ogni classe
    return embeddings.view(embeddings.shape[0], -1)  # Flatten lungo la dimensione degli esempi

def calculate_dissimilarity_matrix(embeddings, device):
    num_classes = embeddings.shape[0]
    # Matrice di dissimilarità vuota
    dissimilarity_matrix = torch.zeros((num_classes, num_classes), device=device)
    
    # Calcola la dissimilarità tra tutte le coppie di classi
    for i in range(num_classes):
        for j in range(i, num_classes):  # Dissimilarità è simmetrica, quindi calcoliamo una volta per ogni coppia
            dissimilarity = 1 - torch.nn.functional.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0), dim=1)
            dissimilarity_matrix[i, j] = dissimilarity
            dissimilarity_matrix[j, i] = dissimilarity  # Dissimilarità è simmetrica
    for i in range(num_classes):
        minimum=1
        min_j=-1
        for j in range(num_classes):
            if j!=i and dissimilarity_matrix[i][j]<minimum:
                minimum=dissimilarity_matrix[i][j]
                min_j=j
        print("La classe ", i, " è più simile alla classe ", min_j, " con dissimilarità ", minimum)
    
    return dissimilarity_matrix

def plot_matrix_with_annotations(matrix, ax, title):
    cax = ax.matshow(matrix, cmap='viridis')  
    ax.set_title(title)
    return cax

def main():
    # Determina se la GPU è disponibile
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Calcola gli embeddings per CIFAR-100
    embeddings = calculate_embeddings("cifar100")
    
    # Converte gli embeddings in float16 e li sposta su CUDA se disponibile
    embeddings = embeddings.to(device)  
    print("Dimensione embeddings:", embeddings.shape)

    # Appiattisce gli embeddings per ciascuna classe
    flattened_embeddings = flatten_class_embeddings(embeddings)
    print("Dimensione flattened_embeddings:", flattened_embeddings.shape)

    # Calcola la matrice di dissimilarità tra le classi
    dissimilarity_matrix = calculate_dissimilarity_matrix(flattened_embeddings, device)
    print("Dimensione dissimilarity_matrix:", dissimilarity_matrix.shape)

    # Crea una figura con un subplot
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))  # 1 riga, 1 colonna
    
    # Primo plot: dissimilarity_matrix
    cax1 = plot_matrix_with_annotations(dissimilarity_matrix.cpu().numpy(), axes, 'Dissimilarity Matrix (CLS)')  # Visualizza la dissimilarità tra le classi
    
    # Aggiungi barre di colore per il subplot
    fig.colorbar(cax1, ax=axes)
    
    # Aggiungi spazio e salva la figura
    plt.tight_layout()
    plt.savefig('dissimilarity_matrix.png')  # Salva l'immagine
    plt.show()  # Mostra la figura

if __name__ == '__main__':
    main()
