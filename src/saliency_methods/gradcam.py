import hydra
import matplotlib.pyplot as plt
import torch.utils.data as data
from torch import nn

from pytorch_grad_cam import GradCAM

from src.models.classifier import Classifier
from src.utils import *


class gradcam_interface:
    def __init__(self, model, device='cpu', **kwargs):
        self.model = model
        self.device = device
        self.kwargs = kwargs

    def generate_saliency(self, input_images, target_class=None, target_layer=None):
        target_layers = get_layer_by_name(self.model, target_layer)

        if isinstance(target_layers, nn.Module):
            target_layers = [target_layers]

        # Ensure gradients are enabled and input_image requires grad
        input_images.requires_grad = True

        with torch.set_grad_enabled(True):
            cam = GradCAM(self.model, target_layers)
            saliency_maps = cam(input_images)

        saliency_tensor = torch.from_numpy(saliency_maps).detach()

        input_images.requires_grad = False

        return saliency_tensor


def get_layer_by_name(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    raise ValueError(f"Layer '{layer_name}' not found in the model.")


# main for testing the interface of GradCAM made for this project
@hydra.main(config_path='../../config', config_name='config', version_base=None)
def main(cfg):
    # Load ORIGINAL model
    model = Classifier(
        cfg.weights_name,
        num_classes=cfg[cfg.dataset.name].n_classes,
        finetune=False)
    weights = os.path.join(cfg.currentDir, cfg.train.save_path, cfg.dataset.name + '_' + cfg.model + '.pth')
    model.load_state_dict(torch.load(weights, map_location=cfg.device))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Load ICUS UNLEARNED model
    model_unlearned_icus = Classifier(
        cfg.weights_name,
        num_classes=cfg[cfg.dataset.name].n_classes,
        finetune=False)
    weights = os.path.join(cfg.currentDir, cfg.train.save_path, cfg.dataset.name + '_forgetting_size_3_icus_' + cfg.model + '.pth')
    model_unlearned_icus.load_state_dict(torch.load(weights, map_location=cfg.device))
    model_unlearned_icus = model_unlearned_icus.to(device).eval()

    # Load SCRUB UNLEARNED model
    model_unlearned_scrub = Classifier(
        cfg.weights_name,
        num_classes=cfg[cfg.dataset.name].n_classes,
        finetune=False)
    weights = os.path.join(cfg.currentDir, cfg.train.save_path, cfg.dataset.name + '_forgetting_size_3_scrub_' + cfg.model + '.pth')
    model_unlearned_scrub.load_state_dict(torch.load(weights, map_location=cfg.device))
    model_unlearned_scrub = model_unlearned_scrub.to(device).eval()

    # Load BADT UNLEARNED model
    model_unlearned_badt = Classifier(
        cfg.weights_name,
        num_classes=cfg[cfg.dataset.name].n_classes,
        finetune=False)
    weights = os.path.join(cfg.currentDir, cfg.train.save_path, cfg.dataset.name + '_forgetting_size_3_badT_' + cfg.model + '.pth')
    model_unlearned_badt.load_state_dict(torch.load(weights, map_location=cfg.device))
    model_unlearned_badt = model_unlearned_badt.to(device).eval()


    from src.datasets.dataset import load_dataset

    # Load test dataset
    data_dir = os.path.join(cfg.currentDir, cfg.dataset.path)
    train, val, test = load_dataset(cfg.dataset.name, data_dir, cfg.dataset.resize)
    dataloader = data.DataLoader(test, batch_size=cfg.train.batch_size, shuffle=True)

    # Initialize the Saliency method
    target_layers_name = cfg.target_layers[cfg.weights_name.split('_Weights')[0]]
    method = gradcam_interface(model, device=cfg.device, reshape_transform=False)
    method_unlearned_icus = gradcam_interface(model_unlearned_icus, device=cfg.device, reshape_transform=False)
    method_unlearned_scrub = gradcam_interface(model_unlearned_scrub, device=cfg.device, reshape_transform=False)
    method_unlearned_badt = gradcam_interface(model_unlearned_badt, device=cfg.device, reshape_transform=False)
    
    batch_cout = 0
    for images, labels in dataloader:
        batch_cout += 1
        images = images.to(device)

        # Get model predictions
        outputs = model(images)
        outputs_unlearned_icus = model_unlearned_icus(images)
        outputs_unlearned_scrub = model_unlearned_scrub(images)
        outputs_unlearned_badt = model_unlearned_badt(images)

        _, preds = torch.max(outputs, 1)
        _, preds_unlearned_icus = torch.max(outputs_unlearned_icus, 1)
        _, preds_unlearned_scrub = torch.max(outputs_unlearned_scrub, 1)
        _, preds_unlearned_badt = torch.max(outputs_unlearned_badt, 1)

        # Generate saliency maps
        saliency_maps = method.generate_saliency(input_images=images, target_layer=target_layers_name)
        saliency_maps_unlearned_icus = method_unlearned_icus.generate_saliency(input_images=images, target_layer=target_layers_name)
        saliency_maps_unlearned_scrub = method_unlearned_scrub.generate_saliency(input_images=images, target_layer=target_layers_name)
        saliency_maps_unlearned_badt = method_unlearned_badt.generate_saliency(input_images=images, target_layer=target_layers_name)

        image_count = 0
        for i in range(images.size(0)):

            image = images[i]
            saliency = saliency_maps[i]
            saliency_unlearned_icus = saliency_maps_unlearned_icus[i]
            saliency_unlearned_scrub = saliency_maps_unlearned_scrub[i]
            saliency_unlearned_badt = saliency_maps_unlearned_badt[i]
            predicted_class = preds[i]
            predicted_class_unlearned_icus = preds_unlearned_icus[i]
            predicted_class_unlearned_scrub = preds_unlearned_scrub[i]
            predicted_class_unlearned_badt = preds_unlearned_badt[i]
            true_class = labels[i]


            # Create figure
            fig, ax = plt.subplots(1, 5, figsize=(10, 2.5))
            ax[0].imshow(image.squeeze().permute(1, 2, 0).cpu().detach().numpy())
            ax[0].axis('off')
            ax[0].set_title(f'True: {true_class}')
            ax[1].imshow(image.squeeze().permute(1, 2, 0).cpu().detach().numpy())
            ax[1].imshow(saliency.squeeze().cpu().detach().numpy(), cmap='jet', alpha=0.4)
            ax[1].axis('off')
            ax[1].set_title(f'Pred: {predicted_class}')
            ax[2].imshow(image.squeeze().permute(1, 2, 0).cpu().detach().numpy())
            ax[2].imshow(saliency_unlearned_scrub.squeeze().cpu().detach().numpy(), cmap='jet', alpha=0.4)
            ax[2].axis('off')
            ax[2].set_title(f'Scrub Pred: {predicted_class_unlearned_scrub}')
            ax[3].imshow(image.squeeze().permute(1, 2, 0).cpu().detach().numpy())
            ax[3].imshow(saliency_unlearned_icus.squeeze().cpu().detach().numpy(), cmap='jet', alpha=0.4)
            ax[3].axis('off')
            ax[3].set_title(f'Icus Pred: {predicted_class_unlearned_icus}')
            ax[4].imshow(image.squeeze().permute(1, 2, 0).cpu().detach().numpy())
            ax[4].imshow(saliency_unlearned_badt.squeeze().cpu().detach().numpy(), cmap='jet', alpha=0.4)
            ax[4].axis('off')
            ax[4].set_title(f'Badt Pred: {predicted_class_unlearned_badt}')
            
            # save the figure in ./outputs/xai/
            # check if the folder exists
            if not os.path.exists('./outputs/xai/'):
                os.makedirs('./outputs/xai/')
            plt.tight_layout()
            plt.savefig(f'./outputs/xai/saliency_{batch_cout}_{image_count}.png')
            plt.close()

            image_count += 1
        


if __name__ == '__main__':
    main()
