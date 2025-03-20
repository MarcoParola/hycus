import torch
import torch.nn as nn
import time


class NegGradLoss(nn.Module):
    """NegGradLoss or negative cross entropy loss, to be applied only on the forget samples"""
    def __init__(self):
        super(NegGradLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        return -self.ce_loss(inputs, targets)


class NegGradPlusLoss(nn.Module):
    """NegGradPlusLoss, cross entropy loss positive on the retain samples and negative on the forget samples"""
    def __init__(self, negative_classes=[]):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.negative_classes = negative_classes

    def forward(self, logits, targets):
        loss = self.criterion(logits, targets)
        negative_mask = torch.tensor([1.0 if t not in self.negative_classes else -1.0 for t in targets], device=logits.device)
        loss = loss * negative_mask
        return loss.mean()

class RandRelabelingLoss(nn.Module):
    """RandRelabelingLoss, cross entropy loss with random relabeling. It receives only forget samples."""
    def __init__(self, num_classes):
        super(RandRelabelingLoss, self).__init__()
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        batch_size = targets.size(0)
        
        # Generate new random labels avoiding original targets
        all_classes = torch.arange(self.num_classes, device=targets.device).repeat(batch_size, 1)
        mask = all_classes != targets.unsqueeze(1)
        available_classes = all_classes[mask].view(batch_size, -1)
        new_targets = available_classes[torch.arange(batch_size), torch.randint(0, self.num_classes - 1, (batch_size,), device=targets.device)]
        loss = self.criterion(logits, new_targets)
        return loss.mean()


