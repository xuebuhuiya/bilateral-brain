# mobilenetv1_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MobileNetV1(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV1, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(32, 64, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(64, 128, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(128, 128, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(128, 256, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(256, 256, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(256, 512, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            *[DepthwiseSeparableConv(512, 512, stride=1) for _ in range(5)],
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(512, 1024, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(1024, 1024, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def mobilenetv1(num_classes):
    model = MobileNetV1(num_classes=num_classes)
    return model
