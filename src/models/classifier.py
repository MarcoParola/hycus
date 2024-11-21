import torch
import torchvision
import numpy as np
import os


class Classifier(torch.nn.Module):

    def __init__(self, weights, num_classes, finetune=False):
        super().__init__()
        assert "." in weights, "Weights must be <MODEL>.<WEIGHTS>"
        weights_cls = weights.split(".")[0]
        weights_name = weights.split(".")[1]
        self.model_name = weights.split("_Weights")[0].lower()
        self.num_classes = num_classes
        weights_cls = getattr(torchvision.models, weights_cls)
        weights = getattr(weights_cls, weights_name)
        self.model = getattr(torchvision.models, self.model_name)(weights=weights)

        if not finetune:
            for param in self.model.parameters():
                param.requires_grad = False

        # method to set the classifier head independently of the model (as head names are different for each model)
        self._set_model_classifier(weights_cls, num_classes)


    def forward(self, x):
        return self.model(x)

    
    def _set_model_classifier(self, weights_cls, num_classes):
        weights_cls = str(weights_cls)
        if "ConvNeXt" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Flatten(1),
                torch.nn.Linear(self.model.classifier[2].in_features, num_classes)
            )
        elif "EfficientNet" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.model.classifier[1].in_features, num_classes)
            )
        elif "MobileNet" in weights_cls or "VGG" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.model.classifier[0].in_features, num_classes),
            )
        elif "DenseNet" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.model.classifier.in_features, num_classes),
            )
        elif "MaxVit" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(self.model.classifier[5].in_features, num_classes),
            )
        elif "ResNet" in weights_cls or "RegNet" in weights_cls or "GoogLeNet" in weights_cls:
            self.model.fc = torch.nn.Sequential(
                torch.nn.Linear(self.model.fc.in_features, num_classes),
            )
        elif "Swin" in weights_cls:
            self.model.head = torch.nn.Sequential(
                torch.nn.Linear(self.model.head.in_features, num_classes)
            )
        elif "ViT" in weights_cls:
            self.model.heads = torch.nn.Sequential(
                torch.nn.Linear(self.model.hidden_dim, num_classes)
            )
        elif "SqueezeNet1_1" in weights_cls or "SqueezeNet1_0" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1)),
                torch.nn.ReLU(),
                torch.nn.AvgPool2d(kernel_size=13, stride=1, padding=0)
            )

    def get_weights(self, nclasses, nlayers):
        pass

    def set_weights(self, distinct, shared, nlayers):
        pass

if __name__ == '__main__':

    model_list = [
        'ResNet50_Weights.IMAGENET1K_V1',
        'EfficientNet_B1_Weights.IMAGENET1K_V1',
        'VGG16_Weights.IMAGENET1K_V1',
        'DenseNet121_Weights.IMAGENET1K_V1',
        'ResNet152_Weights.IMAGENET1K_V2',
        'EfficientNet_B1_Weights.IMAGENET1K_V1',
    ]

    img = torch.randn(2, 3, 256, 256)

    for model_name in model_list:
        print(model_name)
        model = Classifier(model_name, 10)
        print(model(img).shape)