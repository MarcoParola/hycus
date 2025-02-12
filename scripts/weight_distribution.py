import torch
from src.models.classifier import Classifier
from src.models.model import load_model
from matplotlib import pyplot as plt
import numpy as np



def main():
    weight_folder = 'data/weights'
    model_folder = 'checkpoints'

    #model = Classifier('ResNet18_Weights.IMAGENET1K_V1', 10, finetune=True)
    
    model = load_model('ResNet18_Weights.IMAGENET1K_V1', f'{model_folder}/cifar10_resnet.pth')
    model.to('cuda')

    (orig_distinct, orig_shared) = model.get_weights(10, [1,2])
    print(orig_distinct.shape, orig_shared.shape) # torch.Size([10, 513]) torch.Size([1024])
    
    weights_d = []
    weights_s = []

    for i in range(0, 10):
        weight = torch.load(f'{weight_folder}/weights_{i}.pt')
        distinct = weight[:model.model.fc[0].weight.size(1) + 1]
        shared_part = weight[model.model.fc[0].weight.size(1) + 1:]
        weights_d.append(distinct)
        weights_s.append(shared_part)
        print(shared_part.shape, distinct.shape) # torch.Size([1024]) torch.Size([513])

    # move everything to the cpu
    orig_distinct = orig_distinct.cpu().detach().numpy()
    orig_shared = orig_shared.cpu().detach().numpy()
    weights_d = [w.cpu().detach().numpy() for w in weights_d]
    weights_s = [w.cpu().detach().numpy() for w in weights_s]
    weights_d = np.array(weights_d)
    weights_s = np.array(weights_s)

    # plot first distinct weight
    plt.hist(orig_shared, bins=100, alpha=0.5, label='Original')
    
    # aggregate the 10 shared weights into a single array
    orig_shared_mean = np.mean(weights_s, axis=0)
    plt.hist(orig_shared_mean, bins=100, alpha=0.5, label='Trained')
    
    plt.legend()
    plt.show()

    print(orig_distinct.shape, orig_shared.shape)

    for i in range(10):
        plt.hist(orig_distinct[i], bins=70, alpha=0.5, label=f'Distinct {i}')
        plt.hist(weights_d[i], bins=70, alpha=0.5, label=f'Distinct {i}')
        plt.legend()
        plt.show()

    # plot the distribution of original weights against the weights of the trained model
    # do it both for the shared and distinct part
    
    




    

if __name__ == '__main__':
    main()