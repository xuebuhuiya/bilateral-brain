import torch
import torch.nn as nn
import torch.nn.functional as F

# 创建一个单一卷积层，包含64个3x3卷积核，步长为1，填充为1
# 同时创建一个批量归一化层和ReLU激活
conv_layer = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
bn_layer = nn.BatchNorm2d(num_features=64)

# 初始化权重（一般情况下会自动初始化，这里我们用随机数手动初始化）
torch.manual_seed(0)  # 为了重复结果
conv_layer.weight.data = torch.randn(conv_layer.weight.shape)
conv_layer.bias.data = torch.randn(conv_layer.bias.shape)

# 输入数据：1个通道，9x9尺寸的图像（用随机数表示），并添加一个批次维度和通道维度
input_data = torch.randn(1, 1, 9, 9)

# 应用卷积
conv_output = conv_layer(input_data)

# 应用批量归一化
bn_output = bn_layer(conv_output)

# 应用ReLU激活函数
relu_output = F.relu(bn_output)

# 显示结果
# conv_output.shape, bn_output.shape, relu_output.shape, relu_output

print("Convolution Output Shape:", conv_output.shape)
print("BatchNorm Output Shape:", bn_output.shape)
print("ReLU Output Shape:", relu_output.shape)
print("ReLU Output Values:\n", relu_output)
