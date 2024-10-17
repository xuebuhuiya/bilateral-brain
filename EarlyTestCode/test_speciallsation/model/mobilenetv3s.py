# mobilenet_model.py
import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV3Small, self).__init__()
        self.mobilenetv3_small = models.mobilenet_v3_small(pretrained=True)
        # Replace the last fully connected layer
        self.mobilenetv3_small.classifier[3] = nn.Linear(self.mobilenetv3_small.classifier[3].in_features, num_classes)

    def forward(self, x):
        x = self.mobilenetv3_small(x)
        return x

def mobilenetv3_small(num_classes):
    model = MobileNetV3Small(num_classes=num_classes)
    return model
