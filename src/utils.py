import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Subset
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List
from torch.utils.data import DataLoader
import tqdm
import torch.nn as nn
import numpy as np
import omegaconf

def get_save_model_callback(save_path):
    """Returns a ModelCheckpoint callback
    
    save_path: str: save path
    """
    save_model_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=save_path,
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        save_last=True,
    )
    return save_model_callback

def get_early_stopping(patience=10):
    """Returns an EarlyStopping callback

    patience: int: patience
    """
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=patience,
    )
    return early_stopping_callback

# forgetting_set could be a list or a string
def get_forgetting_subset(forgetting_set, n_classes, forgetting_set_size):
    """Returns a forgetting subset

    forgetting_set: str or list: forgetting set
    n_classes: int: number of classes
    forgetting_set_size: int: forgetting set size
    """
    
    if forgetting_set == 'random':
        print('Random forgetting set')
        # return random values using torch
        return torch.randint(0, n_classes, (forgetting_set_size,)).tolist()
    elif isinstance(forgetting_set, omegaconf.listconfig.ListConfig):
        forgetting_set = list(forgetting_set)
        return forgetting_set
    else:
        print('Unknown forgetting set')
    return None
    
        
def get_retain_and_forget_datasets(full_dataset, forgetting_subset, forgetting_set_size):
    
    # Ottieni gli indici di tutte le etichette nel dataset completo
    all_indices = np.arange(len(full_dataset))
    all_labels = np.array([full_dataset[i][1] for i in all_indices])
    
    # Trova gli indici dei campioni da dimenticare
    forget_indices = []
    for class_idx in forgetting_subset:
        forget_indices = np.where(all_labels == class_idx)[0]
    
    # Trova gli indici dei campioni da mantenere
    retain_indices = list(set(all_indices) - set(forget_indices))
    
    # Crea i subset di PyTorch
    forget_dataset = Subset(full_dataset, forget_indices)
    retain_dataset = Subset(full_dataset, retain_indices)
    
    return retain_dataset, forget_dataset, forget_indices



################## ROBA PER SSD COPIATA DA CORRECTIVE MU####################
class ParameterPerturber:
    def __init__(
        self,
        model,
        opt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        parameters=None,
    ):
        self.model = model
        self.opt = opt
        self.device = device
        self.alpha = None
        self.xmin = None

        print(parameters)
        self.lower_bound = parameters["lower_bound"]
        self.exponent = parameters["exponent"]
        self.magnitude_diff = parameters["magnitude_diff"]  # unused
        self.min_layer = parameters["min_layer"]
        self.max_layer = parameters["max_layer"]
        self.forget_threshold = parameters["forget_threshold"] #unused
        self.dampening_constant = parameters["dampening_constant"] #lambda 
        self.selection_weighting = parameters["selection_weighting"] #alpha

    def zerolike_params_dict(self, model: torch.nn) -> Dict[str, torch.Tensor]:
        """
        Taken from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Returns a dict like named_parameters(), with zeroed-out parameter valuse
        Parameters:
        model (torch.nn): model to get param dict from
        Returns:
        dict(str,torch.Tensor): dict of zero-like params
        """
        return dict(
            [
                (k, torch.zeros_like(p, device=p.device))
                for k, p in model.named_parameters()
            ]
        )

    def calc_importance(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Adapated from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Calculate per-parameter, importance
            returns a dictionary [param_name: list(importance per parameter)]
        Parameters:
        DataLoader (DataLoader): DataLoader to be iterated over
        Returns:
        importances (dict(str, torch.Tensor([]))): named_parameters-like dictionary containing list of importances for each parameter
        """
        criterion = nn.CrossEntropyLoss()
        importances = self.zerolike_params_dict(self.model)
        for (x, y, idx) in tqdm.tqdm(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            
            out = self.model(x)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                self.model.named_parameters(), importances.items()
            ):
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))
        return importances

    def modify_weight(
        self,
        original_importance: List[Dict[str, torch.Tensor]],
        forget_importance: List[Dict[str, torch.Tensor]],
    ) -> None:
        """
        Perturb weights based on the SSD equations given in the paper
        Parameters:
        original_importance (List[Dict[str, torch.Tensor]]): list of importances for original dataset
        forget_importance (List[Dict[str, torch.Tensor]]): list of importances for forget sample
        threshold (float): value to multiply original imp by to determine memorization.

        Returns:
        None

        """

        with torch.no_grad():
            for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                self.model.named_parameters(),
                original_importance.items(),
                forget_importance.items(),
            ):
                # Synapse Selection with parameter alpha
                oimp_norm = oimp.mul(self.selection_weighting)
                locations = torch.where(fimp > oimp_norm)

                # Synapse Dampening with parameter lambda
                weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(
                    self.exponent
                )
                update = weight[locations]
                # Bound by 1 to prevent parameter values to increase.
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound
                p[locations] = p[locations].mul(update)


# default values: 
# "dampening_constant" lambda: 1,
# "selection_weighting" alpha: 10 * model_size_scaler,
# model_size_scaler = 1
# if args.net == "ViT":
#     model_size_scaler = 0.5

#We found hyper-parameters using 50
# runs of the TPE search from Optuna (Akiba et al. 2019), for
# values α ∈ [0.1, 100]) and λ ∈ [0.1, 5]. We only conducted
# this search for the Rocket and Veh2 classes. We use λ=1
# and α=10 for all ResNet18 CIFAR tasks. For PinsFaceRecognition, we use α=50 and λ=0.1 due to the much greater
# similarity between classes. ViT also uses λ=1 on all CIFAR
# tasks. We change α=10 to α=5 for slightly improved performance on class and α=25 on sub-class unlearning.
    
def ssd_tuning(
    model,
    forget_train_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device,
):
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1,
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": dampening_constant,
        "selection_weighting": selection_weighting,
    }

    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    pdr = ParameterPerturber(model, optimizer, device, parameters)
    model = model.eval()

    sample_importances = pdr.calc_importance(forget_train_dl)

    original_importances = pdr.calc_importance(full_train_dl)
    pdr.modify_weight(original_importances, sample_importances)
    return model
 

def retrieve_weights(model):
    weights = model.fc.weight.data
    bias = model.fc.bias.data
    return weights, bias


class LinearLR(_LRScheduler):
    """Set the learning rate of each parameter group with a linear
    schedule: :math:`\eta_{t} = \eta_0*(1 - t/T)`, where :math:`\eta_0` is the
    initial lr, :math:`t` is the current epoch or iteration (zero-based) and
    :math:`T` is the total training epochs or iterations. It is recommended to
    use the iteration based calculation if the total number of epochs is small.
    When last_epoch=-1, sets initial lr as lr.
    It is studied in
    `Budgeted Training: Rethinking Deep Neural Network Training Under Resource
     Constraints`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T (int): Total number of training epochs or iterations.
        last_epoch (int): The index of last epoch or iteration. Default: -1.
        
    .. _Budgeted Training\: Rethinking Deep Neural Network Training Under
    Resource Constraints:
        https://arxiv.org/abs/1905.04753
    """

    def __init__(self, optimizer, T, warmup_epochs=100, last_epoch=-1):
        self.T = float(T)
        self.warm_ep = warmup_epochs
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch - self.warm_ep >= 0:
            rate = (1 - ((self.last_epoch-self.warm_ep)/self.T))
        else:
            rate = (self.last_epoch+1)/(self.warm_ep+1)
        return [rate*base_lr for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        return self.get_lr()