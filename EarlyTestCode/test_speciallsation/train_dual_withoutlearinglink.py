import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from itertools import zip_longest
from tqdm import tqdm 
from model.resnet9 import ResNet9

class ResNet9Features(nn.Module):
    def __init__(self, original_model):
        super(ResNet9Features, self).__init__()
        self.prep = original_model.prep
        self.layer1 = original_model.layer1
        self.res1 = original_model.res1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.res2 = original_model.res2
        self.pool = original_model.pool

        # 添加 1x1 卷积层以调整通道数
        self.adjust_channels1 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.adjust_channels2 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.adjust_channels_res2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        
        # 新增卷积层调整拼接后的通道数
        self.adjust_concat1 = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.adjust_concat2 = nn.Conv2d(512, 256, kernel_size=1, bias=False)
        self.adjust_concat_res2 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)

    def forward(self, x, other_features=None):
        prep = self.prep(x)
        layer1 = self.layer1(prep)
        res1 = layer1 + self.res1(layer1)
        
        if other_features is not None:
            other_res1 = self.adjust_channels1(other_features[0])
            res1 = torch.cat((res1, other_res1), dim=1)
            res1 = self.adjust_concat1(res1)
        
        layer2 = self.layer2(res1)
        if other_features is not None:
            other_layer2 = self.adjust_channels2(other_features[1])
            layer2 = torch.cat((layer2, other_layer2), dim=1)
            layer2 = self.adjust_concat2(layer2)
        
        layer3 = self.layer3(layer2)
        res2 = layer3 + self.res2(layer3)
        if other_features is not None:
            other_res2 = self.adjust_channels_res2(other_features[2])
            res2 = torch.cat((res2, other_res2), dim=1)
            res2 = self.adjust_concat_res2(res2)

        pool = self.pool(res2)
        pool = pool.view(pool.size(0), -1)
        return pool, [res1, layer2, res2]



class CombinedModel(nn.Module):
    def __init__(self, coarse_features_model, fine_features_model):
        super(CombinedModel, self).__init__()
        self.coarse_features_model = coarse_features_model
        self.fine_features_model = fine_features_model
        self.fc_coarse = nn.Linear(1024, 20)  # 调整输入维度为1024
        self.fc_fine = nn.Linear(1024, 100)  # 调整输入维度为1024

    
    def forward(self, x_coarse, x_fine):
        coarse_features, coarse_intermediate = self.coarse_features_model(x_coarse)
        fine_features, fine_intermediate = self.fine_features_model(x_fine, other_features=coarse_intermediate)
        # coarse_features, _ = self.coarse_features_model(x_coarse, other_features=fine_intermediate)
        
        combined_features = torch.cat((coarse_features, fine_features), dim=1)
        out_coarse = self.fc_coarse(combined_features)
        out_fine = self.fc_fine(combined_features)
        return out_coarse, out_fine
    
    # def forward(self, x_coarse, x_fine):
    #     # 提取粗略和细致的初级特征
    #     coarse_pool, coarse_features = self.coarse_features_model(x_coarse)
    #     fine_pool, fine_features = self.fine_features_model(x_fine)

    #     # 特征交换：将粗略模型的特征传给细致模型，反之亦然
    #     # 再次调用模型以使用对方的特征进行进一步处理
    #     coarse_final, _ = self.coarse_features_model(x_coarse, other_features=fine_features)
    #     fine_final, _ = self.fine_features_model(x_fine, other_features=coarse_features)

    #     # 特征整合
    #     combined_features = torch.cat((coarse_final, fine_final), dim=1)

    #     # 全连接层分类
    #     out_coarse = self.fc_coarse(combined_features)
    #     out_fine = self.fc_fine(combined_features)

    #     return out_coarse, out_fine

    # def forward(self, x_coarse, x_fine):
    #         coarse_features, coarse_intermediate = self.coarse_features_model(x_coarse)
    #         fine_features, fine_intermediate = self.fine_features_model(x_fine, other_features=coarse_intermediate)
    #         combined_features = torch.cat((coarse_features, fine_features), dim=1)
    #         out_coarse = self.fc_coarse(combined_features)
    #         out_fine = self.fc_fine(combined_features)
    #         return out_coarse, out_fine


def main():
    # 数据预处理和加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_dir = 'datasets_test2/CIFAR100'
    train_coarse_dataset = ImageFolder(root=os.path.join(data_dir, 'train/coarse'), transform=train_transform)
    train_fine_dataset = ImageFolder(root=os.path.join(data_dir, 'train/fine'), transform=train_transform)

    # 加载测试数据
    test_coarse_dataset = ImageFolder(root=os.path.join(data_dir, 'test/coarse'), transform=test_transform)
    test_fine_dataset = ImageFolder(root=os.path.join(data_dir, 'test/fine'), transform=test_transform)

    train_coarse_loader = DataLoader(train_coarse_dataset, batch_size=128, shuffle=True, num_workers=2)
    train_fine_loader = DataLoader(train_fine_dataset, batch_size=128, shuffle=True, num_workers=2)

    test_coarse_loader = DataLoader(test_coarse_dataset, batch_size=100, shuffle=False, num_workers=2)
    test_fine_loader = DataLoader(test_fine_dataset, batch_size=100, shuffle=False, num_workers=2)


    # 加载预训练模型
    coarse_model = ResNet9(num_classes=20).to(device)  # 用于coarse类别的模型
    fine_model = ResNet9(num_classes=100).to(device)  # 用于fine类别的模型

    coarse_model.load_state_dict(torch.load('best_coarse_model.pth', map_location=device))
    fine_model.load_state_dict(torch.load('best_fine_model.pth', map_location=device))

    coarse_features_model = ResNet9Features(coarse_model).to(device)
    fine_features_model = ResNet9Features(fine_model).to(device)
    
    combined_model = CombinedModel(coarse_features_model, fine_features_model).to(device)
    

    # for param in combined_model.coarse_features_model.parameters():
    #     param.requires_grad = True
    # for param in combined_model.fine_features_model.parameters():
    #     param.requires_grad = True

    for param in combined_model.coarse_features_model.parameters():
        param.requires_grad = False
    for param in combined_model.fine_features_model.parameters():
        param.requires_grad = False
        
    # optimizer = optim.Adadelta([
    #     {'params': combined_model.coarse_features_model.parameters()},
    #     {'params': combined_model.fine_features_model.parameters()},
    #     {'params': combined_model.fc_coarse.parameters()},
    #     {'params': combined_model.fc_fine.parameters()},
    # ])

    optimizer = optim.Adam([
    {'params': combined_model.coarse_features_model.parameters()},
    {'params': combined_model.fine_features_model.parameters()},
    {'params': combined_model.fc_coarse.parameters()},
    {'params': combined_model.fc_fine.parameters()},
    ], lr=0.001)

    criterion_coarse = nn.CrossEntropyLoss()
    criterion_fine = nn.CrossEntropyLoss()

    # criterion_coarse = nn.CrossEntropyLoss(weight=coarse_class_weights)
    # criterion_fine = nn.CrossEntropyLoss(weight=fine_class_weights)


    # 训练新的分类头
    num_epochs = 30
    for epoch in range(num_epochs):
        combined_model.train()
        running_loss = 0.0
        for (coarse_data, fine_data) in tqdm(zip_longest(train_coarse_loader, train_fine_loader), total=len(train_coarse_loader), desc=f"Epoch {epoch+1}/{num_epochs}"):
            if coarse_data is None or fine_data is None:
                continue

            coarse_images, coarse_labels = coarse_data
            fine_images, fine_labels = fine_data

            # 确保 coarse_images 和 fine_images 的批次大小相同
            min_batch_size = min(coarse_images.size(0), fine_images.size(0))
            coarse_images = coarse_images[:min_batch_size]
            coarse_labels = coarse_labels[:min_batch_size]
            fine_images = fine_images[:min_batch_size]
            fine_labels = fine_labels[:min_batch_size]

            coarse_images, coarse_labels = coarse_images.to(device), coarse_labels.to(device)
            fine_images, fine_labels = fine_images.to(device), fine_labels.to(device)
            
            optimizer.zero_grad()
            out_coarse, out_fine = combined_model(coarse_images, fine_images)
            
            loss_coarse = criterion_coarse(out_coarse, coarse_labels)
            loss_fine = criterion_fine(out_fine, fine_labels)
            loss = loss_coarse + loss_fine
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_coarse_loader):.4f}')


    # 评估模型
    combined_model.eval()
    with torch.no_grad():
        correct_coarse = 0
        correct_fine = 0
        total_coarse = 0
        total_fine = 0
        for (coarse_data, fine_data) in zip(test_coarse_loader, test_fine_loader):
            if coarse_data is None or fine_data is None:
                continue
            coarse_images, coarse_labels = coarse_data
            fine_images, fine_labels = fine_data

            # 确保 coarse_images 和 fine_images 的批次大小相同
            min_batch_size = min(coarse_images.size(0), fine_images.size(0))
            coarse_images = coarse_images[:min_batch_size]
            coarse_labels = coarse_labels[:min_batch_size]
            fine_images = fine_images[:min_batch_size]
            fine_labels = fine_labels[:min_batch_size]
            
            coarse_images, coarse_labels = coarse_images.to(device), coarse_labels.to(device)
            fine_images, fine_labels = fine_images.to(device), fine_labels.to(device)
            
            out_coarse, out_fine = combined_model(coarse_images, fine_images)  # 同时使用 coarse_images 和 fine_images 进行前向传播
            
            _, predicted_coarse = torch.max(out_coarse.data, 1)
            _, predicted_fine = torch.max(out_fine.data, 1)
            
            total_coarse += coarse_labels.size(0)
            total_fine += fine_labels.size(0)
            correct_coarse += (predicted_coarse == coarse_labels).sum().item()
            correct_fine += (predicted_fine == fine_labels).sum().item()

        print(f'Accuracy of the model on the test images (coarse): {100 * correct_coarse / total_coarse:.2f}%')
        print(f'Accuracy of the model on the test images (fine): {100 * correct_fine / total_fine:.2f}%')

if __name__ == '__main__':
    main()
