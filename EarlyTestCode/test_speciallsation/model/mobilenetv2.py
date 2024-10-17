# mobilenetv2_model.py
import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2, self).__init__()
        self.mobilenetv2 = models.mobilenet_v2(pretrained=True)
        # Replace the last fully connected layer
        self.mobilenetv2.classifier[1] = nn.Linear(self.mobilenetv2.last_channel, num_classes)

    def forward(self, x):
        x = self.mobilenetv2(x)
        return x

def mobilenetv2(num_classes):
    model = MobileNetV2(num_classes=num_classes)
    return model
