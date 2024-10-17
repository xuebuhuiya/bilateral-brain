import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 定义简化的ResNet-9模型
class SimpleResNet9(nn.Module):
    def __init__(self):
        super(SimpleResNet9, self).__init__()
        # 第1层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # 第2层
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # 第3层
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # 第4层
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        # 第5层
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        # 第6层
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        # 第7层
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        # 全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 第9层：全连接层
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        # 第1层
        x = F.relu(self.bn1(self.conv1(x)))
        # 第2层
        x = F.relu(self.bn2(self.conv2(x)))
        # 跳跃连接1
        identity = x
        # 第3层
        x = F.relu(self.bn3(self.conv3(x)))
        # 跳跃连接结果
        x = x + identity
        # 第4层
        x = F.relu(self.bn4(self.conv4(x)))
        # 跳跃连接2
        identity = x
        # 第5层
        x = F.relu(self.bn5(self.conv5(x)))
        # 跳跃连接结果
        x = x + identity
        # 第6层
        x = F.relu(self.bn6(self.conv6(x)))
        # 跳跃连接3
        identity = x
        # 第7层
        x = F.relu(self.bn7(self.conv7(x)))
        # 跳跃连接结果
        x = x + identity
        # 全局平均池化层
        x = self.avgpool(x)
        # 展平特征图
        x = x.view(x.size(0), -1)
        # 第9层：全连接层
        x = self.fc(x)
        return x

# 实例化模型
model = SimpleResNet9()

# 设置随机数种子以便结果可复现
torch.manual_seed(0)

# 步骤1: 生成一个随机的32x32 RGB图片
batch_size = 1  # 可以根据需要调整批处理大小
inputs = torch.randn(batch_size, 3, 32, 32)

# 模拟真实的标签数据，这里我们随机生成
labels = torch.randint(0, 10, (batch_size,))

# 步骤2: 创建损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模式
model.train()

torch.autograd.set_detect_anomaly(True)
# 训练循环
for epoch in range(10):  # 举例只迭代一次
    optimizer.zero_grad()  # 清空梯度
    outputs = model(inputs)  # 前向传播
    loss = criterion(outputs, labels)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
