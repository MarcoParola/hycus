from torchvision.datasets import CIFAR100

# Carica il dataset CIFAR-100
dataset = CIFAR100(root="./data", download=True)

# Stampa l'elenco delle classi
classes = dataset.classes
print(classes)