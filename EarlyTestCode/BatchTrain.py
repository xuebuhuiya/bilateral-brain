# illustrate an example of training a neural network with a batch size of 512.
# simulate a training dataset with 10240 images, where each image is represented as a 32x32 pixel with 3 color channels (RGB).
# use a simple neural network for the example.

import torch
from torch.utils.data import DataLoader, TensorDataset

# 模拟CIFAR-100训练数据集
# 10240张图片，每张图片有3个通道，尺寸为32x32像素
num_samples = 10240
num_channels = 3
image_height = 32
image_width = 32

# 生成随机图片和随机标签（100个类别）
images = torch.randn(num_samples, num_channels, image_height, image_width)
labels = torch.randint(0, 100, (num_samples,))

# 创建数据集和数据加载器
train_dataset = TensorDataset(images, labels)
train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)

# 定义一个简单的卷积神经网络
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        # 将卷积层输出平铺后的向量连接到一个有100个输出的全连接层
        self.fc1 = torch.nn.Linear(64 * image_height * image_width, 100)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # 应用ReLU激活函数
        x = x.view(x.size(0), -1)  # 将张量展平
        x = self.fc1(x)  # 应用全连接层
        return x

# 初始化神经网络
model = SimpleCNN()

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 使用Adam优化器

# 模拟一轮训练
for epoch in range(1):  # 遍历数据集多次（本例中为1轮）
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # 获取输入数据；data是[输入, 标签]的列表
        inputs, labels = data

        # 清零参数梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        
        # 更新权重
        optimizer.step()
        
        # 打印统计信息
        running_loss += loss.item()
        if i % 20 == 19:    # 每20个小批量打印一次
            print(f'批次: {i+1}, 损失: {running_loss / 20}')
            running_loss = 0.0

print('Finished Training for one epoch')
