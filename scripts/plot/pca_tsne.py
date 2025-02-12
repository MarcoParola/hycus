import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import hydra
from datetime import datetime
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.datasets.dataset import load_dataset
from src.models.classifier import Classifier


def plot_features_3d(cfg, model, data_loader, pca=None, unlearned=False): 
    model.eval()  

    all_features = []
    all_labels = []

    forgetting_subset = cfg.forgetting_set

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
        ax.legend(loc='best')  

    # Save the plot with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if unlearned:
        if cfg.unlearning_method=="icus":
            plt.savefig('fig/feature_plot_3d_unlearned_forgetting_set_'+str(forgetting_subset)+'_'+cfg.unlearning_method+'_'+cfg.unlearn.aggregation_method+'.png')
        else:
            plt.savefig('fig/feature_plot_3d_unlearned_forgetting_set_'+str(forgetting_subset)+'_'+cfg.unlearning_method+'.png')
    else:
        if cfg.golden_model==False:
            if cfg.unlearning_method=="icus":
                plt.savefig('fig/feature_plot_3d_unlearned_forgetting_set_'+str(forgetting_subset)+'_'+cfg.unlearning_method+'_'+cfg.unlearn.aggregation_method+'.png')
            else:
                plt.savefig('fig/feature_plot_3d_forgetting_set_'+str(forgetting_subset)+'_'+cfg.unlearning_method+'.png')
        else:
            plt.savefig('fig/feature_plot_3d_unlearned_forgetting_set_'+str(forgetting_subset)+'_golden.png')
    # Show the plot
    plt.show()
    if not unlearned:
        return pca


def plot_features(cfg, model, data_loader, pca=None, unlearned=False, shared_limits=None): 
    model.eval()  

    all_features = []
    all_labels = []

    forgetting_subset = cfg.forgetting_set

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
        
        ax.set_xlim(shared_limits["tsne"][x_comp])
        ax.set_ylim(shared_limits["tsne"][y_comp])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not unlearned:
        plt.savefig('fig/feature_plot_2d_original_'+cfg.dataset.name+'.png')
    else:
        if cfg.golden_model==False:
            plt.savefig('fig/feature_plot_2d_forgetting_set_'+str(forgetting_subset)+'_'+cfg.unlearning_method+'.png')
        else:
            plt.savefig('fig/feature_plot_2d_forgetting_set_'+str(forgetting_subset)+'_golden.png')
    plt.show()

    if not unlearned:
        return pca, shared_limits
    else:
        return None

@hydra.main(config_path='../../config', config_name='config')
def main(cfg):
    os.chdir('../../..')
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    _, _, test = load_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)
    
    # Data loaders
    test_loader = torch.utils.data.DataLoader(test, 
        batch_size=cfg.train.batch_size, 
        shuffle=False, 
        num_workers=cfg.train.num_workers)
    model = Classifier(cfg.weights_name, num_classes=cfg[cfg.dataset.name].n_classes, finetune=True)
    model.to(cfg.device)
    weights = os.path.join(cfg.currentDir, cfg.train.save_path, cfg.dataset.name + '_' + cfg.model + '.pth')
    model.load_state_dict(torch.load(weights, map_location=cfg.device))
    pca, shared_limits = plot_features(cfg, model, test_loader, unlearned=False)
    
    if cfg.golden_model==True:
        weights = os.path.join(cfg.currentDir, cfg.train.save_path, cfg.dataset.name +'_resnet_only_retain_set'+str(cfg.forgetting_set)+ '.pth')
        model.load_state_dict(torch.load(weights, map_location=cfg.device))
        plot_features(cfg,model, test_loader, pca=pca, unlearned=True, shared_limits=shared_limits)
        
    elif cfg.original_model==False:
        weights = os.path.join(cfg.currentDir, cfg.train.save_path, cfg.dataset.name + '_forgetting_set_'+str(cfg.forgetting_set) +'_'+cfg.unlearning_method+ '_resnet.pth')
        model.load_state_dict(torch.load(weights, map_location=cfg.device))
        plot_features(cfg,model, test_loader, pca=pca, unlearned=True, shared_limits=shared_limits)
        

if __name__ == "__main__":
    main()