# vgg11_bn_model.py
import torch
import torch.nn as nn
import torchvision.models as models

class VGG11_BN(nn.Module):
    def __init__(self, num_classes):
        super(VGG11_BN, self).__init__()
        self.vgg11_bn = models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1)
        # Replace the last fully connected layer
        in_features = self.vgg11_bn.classifier[6].in_features
        self.vgg11_bn.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.vgg11_bn(x)
        return x

def vgg11_bn(num_classes):
    model = VGG11_BN(num_classes=num_classes)
    return model