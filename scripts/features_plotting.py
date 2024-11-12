import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from datetime import datetime

def plot_features(model, data_loader, forgetting_subset, unlearned=False): 
    model.eval()  

    all_features = []
    all_labels = []

    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  

    with torch.no_grad():
        for x, y in data_loader:  
            x, y = x.to(device), y.to(device)  
            features = model.extract_features(x)  
            all_features.append(features.cpu())  
            all_labels.append(y.cpu())  

    features_np = torch.cat(all_features).numpy()
    labels_np = torch.cat(all_labels).numpy()

    # PCA transformation
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(features_np)

    # t-SNE transformation
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(features_np)

    # Plotting PCA
    plt.figure(figsize=(14, 6))

    # Plot PCA
    plt.subplot(1, 2, 1)
    for classe in np.unique(labels_np):
        if classe in forgetting_subset:
            indices = np.where(labels_np == classe)
            plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=f'Class forget {classe}',alpha=0.4, s=2)
        else:
            indices = np.where(labels_np == classe)
            plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=f'Class {classe}', alpha=0.4, s=2)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of extracted features from the model')
    plt.legend()
    plt.grid(True)

    # Plot t-SNE
    plt.subplot(1, 2, 2)
    for classe in np.unique(labels_np):
        if classe in forgetting_subset:
            indices = np.where(labels_np == classe)
            plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=f'Class forget {classe}', alpha=0.4, s=2)
        else:
            indices = np.where(labels_np == classe)
            plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=f'Class {classe}', alpha=0.4, s=2)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE of extracted features from the model')
    plt.legend()
    plt.grid(True)

    # Save the plot with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if unlearned:
        plt.savefig('fig/feature_plot_unlearned_'+timestamp+'.png')
    else:
        plt.savefig('fig/feature_plot_'+timestamp+'.png')
    plt.show()


def plot_features_3d(model, data_loader, forgetting_subset, unlearned=False): 
    model.eval()  

    all_features = []
    all_labels = []

    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  

    with torch.no_grad():
        for x, y in data_loader:  
            x, y = x.to(device), y.to(device)  
            features = model.extract_features(x)  
            all_features.append(features.cpu())  
            all_labels.append(y.cpu())  

    features_np = torch.cat(all_features).numpy()
    labels_np = torch.cat(all_labels).numpy()

    # PCA transformation to 3D
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(features_np)

    # t-SNE transformation to 3D
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(features_np)

    # Plotting in 3D
    fig = plt.figure(figsize=(14, 6))

    # Plot PCA in 3D
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    for classe in np.unique(labels_np):
        if classe in forgetting_subset:
            indices = np.where(labels_np == classe)
            ax1.scatter(X_pca[indices, 0], X_pca[indices, 1], X_pca[indices, 2], label=f'Class forget {classe}' ,alpha=0.4, s=2)
        else:
            indices = np.where(labels_np == classe)
            ax1.scatter(X_pca[indices, 0], X_pca[indices, 1], X_pca[indices, 2], label=f'Class {classe}', alpha=0.4, s=2)
    ax1.set_xlabel('PCA Component 1')
    ax1.set_ylabel('PCA Component 2')
    ax1.set_zlabel('PCA Component 3')
    ax1.set_title('3D PCA of extracted features from the model')
    ax1.legend()
    ax1.grid(True)

    # Plot t-SNE in 3D
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    for classe in np.unique(labels_np):
        if classe in forgetting_subset:
            indices = np.where(labels_np == classe)
            ax2.scatter(X_tsne[indices, 0], X_tsne[indices, 1], X_tsne[indices, 2], label=f'Class forget {classe}',alpha=0.4, s=2)
        else:
            indices = np.where(labels_np == classe)
            ax2.scatter(X_tsne[indices, 0], X_tsne[indices, 1], X_tsne[indices, 2], label=f'Class {classe}', alpha=0.4, s=2)
    ax2.set_xlabel('t-SNE Component 1')
    ax2.set_ylabel('t-SNE Component 2')
    ax2.set_zlabel('t-SNE Component 3')
    ax2.set_title('3D t-SNE of extracted features from the model')
    ax2.legend()
    ax2.grid(True)

    # Save the plot with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if unlearned:
        plt.savefig('fig/feature_plot_3d_unlearned_'+timestamp+'.png')
    else:
        plt.savefig('fig/feature_plot_3d_'+timestamp+'.png')
    plt.show()


