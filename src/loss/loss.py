import torch
import torch.nn as nn


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