import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from datetime import datetime

"""
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
    plt.subplot(1, 2, 1) #2,3,1
    for classe in np.unique(labels_np):
        if classe in forgetting_subset:
            indices = np.where(labels_np == classe)
            plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=f'Class forget {classe}',alpha=0.4, s=10)
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
            plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=f'Class forget {classe}', alpha=0.4, s=10)
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
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
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
"""

def plot_features_3d(model, data_loader, forgetting_subset, pca=None, unlearned=False): 
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
    if not unlearned:
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(features_np)
    else:
        X_pca = pca.transform(features_np)

    # t-SNE transformation to 3D
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(features_np)

    # Create figure with 6 subplots (3 for PCA, 3 for t-SNE)
    fig = plt.figure(figsize=(18, 12))

    pca_components = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]  

    for i, (x_comp, y_comp, z_comp) in enumerate(pca_components, start=1):
        ax = fig.add_subplot(2, 3, i, projection='3d')
        for classe in np.unique(labels_np):
            if classe in forgetting_subset:
                indices = np.where(labels_np == classe)
                ax.scatter(X_pca[indices, x_comp], X_pca[indices, y_comp], X_pca[indices, z_comp], label=f'Class forget {classe}', alpha=0.4, s=10)
            else:
                indices = np.where(labels_np == classe)
                ax.scatter(X_pca[indices, x_comp], X_pca[indices, y_comp], X_pca[indices, z_comp], label=f'Class {classe}', alpha=0.4, s=2)
        ax.set_xlabel(f'PCA Component {x_comp + 1}')
        ax.set_ylabel(f'PCA Component {y_comp + 1}')
        ax.set_zlabel(f'PCA Component {z_comp + 1}')
        ax.set_title(f'PCA 3D - View {i}')
        ax.grid(True)
        ax.legend(loc='best')  

    # Plot t-SNE in 3D from 3 different angles
    tsne_components = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]  # Change component distribution for t-SNE

    for i, (x_comp, y_comp, z_comp) in enumerate(tsne_components, start=4):
        ax = fig.add_subplot(2, 3, i, projection='3d')
        for classe in np.unique(labels_np):
            if classe in forgetting_subset:
                indices = np.where(labels_np == classe)
                ax.scatter(X_tsne[indices, x_comp], X_tsne[indices, y_comp], X_tsne[indices, z_comp], label=f'Class forget {classe}', alpha=0.4, s=10)
            else:
                indices = np.where(labels_np == classe)
                ax.scatter(X_tsne[indices, x_comp], X_tsne[indices, y_comp], X_tsne[indices, z_comp], label=f'Class {classe}', alpha=0.4, s=2)
        ax.set_xlabel(f't-SNE Component {x_comp + 1}')
        ax.set_ylabel(f't-SNE Component {y_comp + 1}')
        ax.set_zlabel(f't-SNE Component {z_comp + 1}')
        ax.set_title(f't-SNE 3D - View {i}')
        ax.grid(True)
        ax.legend(loc='best')  # Aggiungi la legenda qui

    # Save the plot with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if unlearned:
        plt.savefig('fig/feature_plot_3d_unlearned_'+timestamp+'.png')
    else:
        plt.savefig('fig/feature_plot_3d_'+timestamp+'.png')
    
    # Show the plot
    plt.show()
    if not unlearned:
        return pca


def plot_features(model, data_loader, forgetting_subset, pca=None, unlearned=False, shared_limits=None): 
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

    if not unlearned:
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(features_np)  
    else:
        X_pca = pca.transform(features_np)  
    
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(features_np)  
    
    if shared_limits is None:
        shared_limits = {"pca": {}, "tsne": {}}
        for i in range(3):  # Per PCA e t-SNE
            shared_limits["pca"][i] = (np.min(X_pca[:, i]), np.max(X_pca[:, i]))
            shared_limits["tsne"][i] = (np.min(X_tsne[:, i]), np.max(X_tsne[:, i]))
    
    plt.figure(figsize=(18, 12))
    component_pairs = [(0, 1), (0, 2), (1, 2)]  

    for i, (x_comp, y_comp) in enumerate(component_pairs, start=1):
        ax = plt.subplot(2, 3, i)  
        for classe in np.unique(labels_np):
            if classe in forgetting_subset:
                indices = np.where(labels_np == classe)
                ax.scatter(X_pca[indices, x_comp], X_pca[indices, y_comp], label=f'Class forget {classe}', alpha=0.4, s=10)
            else:
                indices = np.where(labels_np == classe)
                ax.scatter(X_pca[indices, x_comp], X_pca[indices, y_comp], label=f'Class {classe}', alpha=0.4, s=2)
        ax.set_xlabel(f'PCA Component {x_comp + 1}')
        ax.set_ylabel(f'PCA Component {y_comp + 1}')
        ax.set_title(f'PCA: Component {x_comp + 1} vs Component {y_comp + 1}')
        ax.grid(True)
        ax.legend(loc='best')
        # Imposta i limiti PCA
        ax.set_xlim(shared_limits["pca"][x_comp])
        ax.set_ylim(shared_limits["pca"][y_comp])

    for i, (x_comp, y_comp) in enumerate(component_pairs, start=4):
        ax = plt.subplot(2, 3, i)  
        for classe in np.unique(labels_np):
            if classe in forgetting_subset:
                indices = np.where(labels_np == classe)
                ax.scatter(X_tsne[indices, x_comp], X_tsne[indices, y_comp], label=f'Class forget {classe}', alpha=0.4, s=10)
            else:
                indices = np.where(labels_np == classe)
                ax.scatter(X_tsne[indices, x_comp], X_tsne[indices, y_comp], label=f'Class {classe}', alpha=0.4, s=2)
        ax.set_xlabel(f't-SNE Component {x_comp + 1}')
        ax.set_ylabel(f't-SNE Component {y_comp + 1}')
        ax.set_title(f't-SNE: Component {x_comp + 1} vs Component {y_comp + 1}')
        ax.grid(True)
        ax.legend(loc='best')
        # Imposta i limiti t-SNE
        ax.set_xlim(shared_limits["tsne"][x_comp])
        ax.set_ylim(shared_limits["tsne"][y_comp])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if unlearned:
        plt.savefig('fig/feature_plot_2d_unlearned_'+timestamp+'.png')
    else:
        plt.savefig('fig/feature_plot_2d_'+timestamp+'.png')
    plt.show()

    if not unlearned:
        return pca, shared_limits
    else:
        return None
