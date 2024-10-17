import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import os

if __name__ == "__main__":
    # 数据准备
    data_dir = 'datasets_test2/CIFAR100'
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # 加载训练数据
    train_coarse_dataset = ImageFolder(root=os.path.join(data_dir, 'train/coarse'), transform=train_transform)
    train_fine_dataset = ImageFolder(root=os.path.join(data_dir, 'train/fine'), transform=train_transform)

    # 加载测试数据
    test_coarse_dataset = ImageFolder(root=os.path.join(data_dir, 'test/coarse'), transform=test_transform)
    test_fine_dataset = ImageFolder(root=os.path.join(data_dir, 'test/fine'), transform=test_transform)

    train_coarse_loader = DataLoader(train_coarse_dataset, batch_size=128, shuffle=True, num_workers=2)
    train_fine_loader = DataLoader(train_fine_dataset, batch_size=128, shuffle=True, num_workers=2)

    test_coarse_loader = DataLoader(test_coarse_dataset, batch_size=100, shuffle=False, num_workers=2)
    test_fine_loader = DataLoader(test_fine_dataset, batch_size=100, shuffle=False, num_workers=2)

    # 模型设计
    class MultiTaskModel(nn.Module):
        def __init__(self):
            super(MultiTaskModel, self).__init__()
            self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.base_model.fc = nn.Identity()  # 移除最后一层
            self.fc_coarse = nn.Linear(512, 20)  # 大类分类层
            self.fc_fine = nn.Linear(512, 100)  # 小类分类层

        def forward(self, x):
            x = self.base_model(x)
            coarse_output = self.fc_coarse(x)
            fine_output = self.fc_fine(x)
            return coarse_output, fine_output

    model = MultiTaskModel()
    criterion_coarse = nn.CrossEntropyLoss()
    criterion_fine = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    # 训练模型
    def train(model, train_coarse_loader, train_fine_loader, optimizer, criterion_coarse, criterion_fine, device):
        model.train()
        for (data_coarse, target_coarse), (data_fine, target_fine) in zip(train_coarse_loader, train_fine_loader):
            data_coarse, target_coarse = data_coarse.to(device), target_coarse.to(device)
            data_fine, target_fine = data_fine.to(device), target_fine.to(device)
            optimizer.zero_grad()
            coarse_output, fine_output = model(data_coarse)
            loss_coarse = criterion_coarse(coarse_output, target_coarse)
            fine_output, fine_output = model(data_fine)
            loss_fine = criterion_fine(fine_output, target_fine)
            loss = loss_coarse + loss_fine
            loss.backward()
            optimizer.step()

    # 评估模型
    def evaluate(model, test_coarse_loader, test_fine_loader, device):
        model.eval()
        correct_coarse = 0
        correct_fine = 0
        total = 0
        with torch.no_grad():
            for (data_coarse, target_coarse), (data_fine, target_fine) in zip(test_coarse_loader, test_fine_loader):
                data_coarse, target_coarse = data_coarse.to(device), target_coarse.to(device)
                data_fine, target_fine = data_fine.to(device), target_fine.to(device)
                coarse_output, fine_output = model(data_coarse)
                _, predicted_coarse = torch.max(coarse_output, 1)
                _, predicted_fine = torch.max(fine_output, 1)
                total += target_coarse.size(0)
                correct_coarse += (predicted_coarse == target_coarse).sum().item()
                correct_fine += (predicted_fine == target_fine).sum().item()
                # 获取 top-5 概率
                probs_coarse = F.softmax(coarse_output, dim=1)
                probs_fine = F.softmax(fine_output, dim=1)
                top5_coarse_probs, top5_coarse_classes = probs_coarse.topk(5, dim=1)
                top5_fine_probs, top5_fine_classes = probs_fine.topk(5, dim=1)
                print(f'Top-5 Coarse Classes: {top5_coarse_classes}')
                print(f'Top-5 Fine Classes: {top5_fine_classes}')
        print(f'Coarse Accuracy: {100 * correct_coarse / total:.2f}%')
        print(f'Fine Accuracy: {100 * correct_fine / total:.2f}%')

    # 训练和评估
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(10):
        train(model, train_coarse_loader, train_fine_loader, optimizer, criterion_coarse, criterion_fine, device)
        evaluate(model, test_coarse_loader, test_fine_loader, device)
