import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_channels, growth_rate):
        layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
        )
        return layer

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.layer(x)

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, num_classes=100):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 2 * growth_rate, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2 * growth_rate),
            nn.ReLU(inplace=True),
        )

        self.block1 = DenseBlock(2 * growth_rate, growth_rate, 6)
        self.trans1 = TransitionLayer(2 * growth_rate + 6 * growth_rate, growth_rate)

        self.block2 = DenseBlock(growth_rate, growth_rate, 12)
        self.trans2 = TransitionLayer(growth_rate + 12 * growth_rate, growth_rate)

        self.block3 = DenseBlock(growth_rate, growth_rate, 24)
        self.trans3 = TransitionLayer(growth_rate + 24 * growth_rate, growth_rate)

        self.block4 = DenseBlock(growth_rate, growth_rate, 16)

        self.bn = nn.BatchNorm2d(growth_rate + 16 * growth_rate)
        self.fc = nn.Linear(growth_rate + 16 * growth_rate, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.trans3(self.block3(x))
        x = self.block4(x)
        x = torch.mean(x, dim=[2, 3])
        x = self.fc(x)
        return x

# # Example usage:
# model = DenseNet(num_classes=10)
