import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
# 定义一个简单的全连接神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 128)  # 假设输入是32x32 RGB图像，所以3个颜色通道
        self.fc2 = nn.Linear(128, 10)  # 假设是一个10类分类问题

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)  # 展平图像
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型
model = SimpleNN()

# 使用随机数据和标签来模拟一个批次的输入
batch_size = 512
inputs = torch.randn(batch_size, 3, 32, 32)  # 创建一个形状为[512, 3, 32, 32]的随机张量来模拟输入图像
targets = torch.randint(0, 10, (batch_size,))  # 创建一个随机整数的张量来模拟目标标签

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练的一个批次
optimizer.zero_grad()   # 清空之前的梯度
outputs = model(inputs)  # 前向传播
loss = criterion(outputs, targets)  # 计算损失
loss.backward()  # 反向传播
optimizer.step()  # 更新权重

# 输出损失值
print(loss.item())