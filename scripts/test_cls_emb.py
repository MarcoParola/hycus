import requests
from transformers import BertModel, BertTokenizer
import torch
from descr_and_similarity import calculate_embeddings, calculate_dissimilarity
import matplotlib.pyplot as plt
import numpy as np  
from sklearn.metrics import pairwise_distances

def flatten_class_embeddings(embeddings):
    # flatting the embeddings
    return embeddings.view(embeddings.shape[0], -1)  

def calculate_dissimilarity_matrix(embeddings, device):
    num_classes = embeddings.shape[0]
    # Empty matrix to store the dissimilarity between all pairs of classes
    dissimilarity_matrix = torch.zeros((num_classes, num_classes), device=device)
    
    # calculate the dissimilarity between all pairs of classes
    for i in range(num_classes):
        for j in range(i, num_classes):  
            dissimilarity = 1 - torch.nn.functional.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0), dim=1)
            dissimilarity_matrix[i, j] = dissimilarity
            dissimilarity_matrix[j, i] = dissimilarity  # dissimilarity_matrix is symmetric
    for i in range(num_classes):
        minimum=1
        min_j=-1
        for j in range(num_classes):
            if j!=i and dissimilarity_matrix[i][j]<minimum:
                minimum=dissimilarity_matrix[i][j]
                min_j=j
    
    return dissimilarity_matrix

def plot_matrix_with_annotations(matrix, ax, title):
    cax = ax.matshow(matrix, cmap='viridis')  
    ax.set_title(title)
    return cax

def main():
    # check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # embeddings for dataset
    embeddings = calculate_embeddings("cifar100")
    embeddings = embeddings.to(device)  

    # flatten the embeddings
    flattened_embeddings = flatten_class_embeddings(embeddings)

    # calculate the dissimilarity matrix
    dissimilarity_matrix = calculate_dissimilarity_matrix(flattened_embeddings, device)

    # Plot the dissimilarity matrix
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))  # 1 riga, 1 colonna
    cax1 = plot_matrix_with_annotations(dissimilarity_matrix.cpu().numpy(), axes, 'Dissimilarity Matrix (CLS)')  # Visualizza la dissimilarità tra le classi
    fig.colorbar(cax1, ax=axes)
    plt.tight_layout()
    plt.savefig('dissimilarity_matrix.png')  # Save image
    plt.show()  # show image

if __name__ == '__main__':
    main()
