import torch
import torchvision
import numpy as np
import os


class Classifier(torch.nn.Module):

    def __init__(self, weights, num_classes, finetune=False):
        super().__init__()
        assert "." in weights, "Weights must be <MODEL>.<WEIGHTS>"
        self.weights_cls = weights
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

    def extract_features(self, x):
        if "DenseNet" in self.weights_cls:
            features = self.model.features(x)
            out = nn.ReLU(inplace=True)(features)
            out = nn.AdaptiveAvgPool2d((1, 1))(out)
            out = torch.flatten(out, 1)
        elif "MaxVit" in self.weights_cls:
            features = self.model.forward_features(x)
        elif "ResNet" in self.weights_cls or "RegNet" in self.weights_cls or "GoogLeNet" in self.weights_cls:
            features = self.model.avgpool(self.model.layer4(self.model.layer3(self.model.layer2(self.model.layer1(self.model.relu(self.model.bn1(self.model.conv1(x))))))))
            features = torch.flatten(features, 1)
        elif "Swin" in self.weights_cls:
            features = self.model.forward_features(x)
        elif "ViT" in self.weights_cls:
            features = self.model.forward_features(x)
        elif "SqueezeNet1_1" in self.weights_cls or "SqueezeNet1_0" in self.weights_cls:
            features = self.model.features(x)
        else:
            raise ValueError("Unsupported model type")
        return features

    def get_weights(self, nclasses, nlayers):
        shared = torch.empty(0)
        distinct = torch.empty(0)
        if torch.cuda.is_available():
            shared = shared.to('cuda')
            distinct = distinct.to('cuda')

        if self.model_name == 'resnet18':
            for l in nlayers:
                if l == 1:
                    w = self.model.fc[0].weight.data  
                    bias = self.model.fc[0].bias.data  
                    bias = bias.unsqueeze(1) 

                    if distinct.numel() == 0: 
                        distinct = torch.cat([w, bias], dim=1)  
                    else:
                        distinct = torch.cat([distinct, w, bias], dim=1)

                elif l==2:
                    shared=torch.cat((shared, self.model.layer4[1].bn2.weight.data.view(-1)))
                    shared=torch.cat((shared, self.model.layer4[1].bn2.bias.data.view(-1)))

                elif l==3:
                    shared=torch.cat((shared, self.model.layer4[1].conv2.weight.data.view(-1)))

                elif l==4:
                    shared=torch.cat((shared, self.model.layer2[1].bn1.weight.data.view(-1)))
                    shared=torch.cat((shared, self.model.layer2[1].bn1.bias.data.view(-1)))

                elif l==5:
                    shared=torch.cat((shared, self.model.layer1[1].conv1.weight.data.view(-1)))

        elif self.model_name == 'efficientnet_b0':
            pass
        
        return (distinct, shared)


    def set_weights(self, distinct, shared, nclasses, nlayers):
        idx_distinct = 0
        idx_shared = 0

        if torch.cuda.is_available():
            distinct = distinct.to('cuda')
            shared = shared.to('cuda')

        if self.model_name == 'resnet18':
            for l in nlayers:
                if l == 1:
                    for i in range(distinct.size(0)):
                        self.model.fc[0].weight.data[i] = distinct[i][:-1]
                        self.model.fc[0].bias.data[i] = distinct[i][-1]
                elif l == 2:
                    # Assegna i pesi e bias a layer4[1].bn2
                    bn_w_shape = self.model.layer4[1].bn2.weight.data.shape
                    bn_b_shape = self.model.layer4[1].bn2.bias.data.shape

                    bn_weight = shared[idx_shared:idx_shared + bn_w_shape[0]].view(bn_w_shape)
                    bn_bias = shared[idx_shared + bn_w_shape[0]:idx_shared + bn_w_shape[0] + bn_b_shape[0]].view(bn_b_shape)

                    self.model.layer4[1].bn2.weight.data.copy_(bn_weight)
                    self.model.layer4[1].bn2.bias.data.copy_(bn_bias)

                    idx_shared += bn_w_shape[0] + bn_b_shape[0]

                elif l == 3:
                    # Assegna i pesi a layer4[1].conv2
                    conv_w_shape = self.model.layer4[1].conv2.weight.data.shape
                    conv_weight = shared[idx_shared:idx_shared + conv_w_shape[0] * conv_w_shape[1] * conv_w_shape[2] * conv_w_shape[3]].view(conv_w_shape)

                    self.model.layer4[1].conv2.weight.data.copy_(conv_weight)

                    idx_shared += conv_w_shape[0] * conv_w_shape[1] * conv_w_shape[2] * conv_w_shape[3]

                elif l == 4:
                    # Assegna i pesi e bias a layer2[1].bn1
                    bn_w_shape = self.model.layer2[1].bn1.weight.data.shape
                    bn_b_shape = self.model.layer2[1].bn1.bias.data.shape

                    bn_weight = shared[idx_shared:idx_shared + bn_w_shape[0]].view(bn_w_shape)
                    bn_bias = shared[idx_shared + bn_w_shape[0]:idx_shared + bn_w_shape[0] + bn_b_shape[0]].view(bn_b_shape)

                    self.model.layer2[1].bn1.weight.data.copy_(bn_weight)
                    self.model.layer2[1].bn1.bias.data.copy_(bn_bias)

                    idx_shared += bn_w_shape[0] + bn_b_shape[0]

                elif l == 5:
                    # Assegna i pesi a layer1[1].conv1
                    conv_w_shape = self.model.layer1[1].conv1.weight.data.shape
                    conv_weight = shared[idx_shared:idx_shared + conv_w_shape[0] * conv_w_shape[1] * conv_w_shape[2] * conv_w_shape[3]].view(conv_w_shape)

                    self.model.layer1[1].conv1.weight.data.copy_(conv_weight)

                    idx_shared += conv_w_shape[0] * conv_w_shape[1] * conv_w_shape[2] * conv_w_shape[3]

        elif self.model_name == 'efficientnet_b0':
            # Se hai bisogno di implementare anche per EfficientNet B0, puoi farlo qui
            pass

        return



if __name__ == '__main__':

    model_list = [
        # 'ResNet50_Weights.IMAGENET1K_V1',
        # 'EfficientNet_B1_Weights.IMAGENET1K_V1',
        # 'VGG16_Weights.IMAGENET1K_V1',
        # 'DenseNet121_Weights.IMAGENET1K_V1',
        # 'ResNet152_Weights.IMAGENET1K_V2',
        # 'EfficientNet_B1_Weights.IMAGENET1K_V1',
        'ResNet18_Weights.IMAGENET1K_V1',
        'EfficientNet_B0_Weights.IMAGENET1K_V1'
    ]

    img = torch.randn(2, 3, 256, 256)

    for model_name in model_list:
        print(model_name)
        model = Classifier(model_name, 10)
        print(model.model_name, model(img).shape, model.extract_feature(img).shape)
        
        