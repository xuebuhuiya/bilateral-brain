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
import numpy as np
import pandas as pd
import random

# 定义ResNet9特征提取模型
class ResNet9Features(nn.Module):
    def __init__(self, original_model):
        super(ResNet9Features, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

# 定义合并模型
class CombinedModel(nn.Module):
    def __init__(self, coarse_features_model, fine_features_model):
        super(CombinedModel, self).__init__()
        self.coarse_features_model = coarse_features_model
        self.fine_features_model = fine_features_model
        self.fc_coarse = nn.Linear(8192*2, 20)  
        self.fc_fine = nn.Linear(8192*2, 100)

    def forward(self, x_coarse, x_fine):
        coarse_features = self.coarse_features_model(x_coarse)
        fine_features = self.fine_features_model(x_fine)
        combined_features = torch.cat((coarse_features, fine_features), dim=1)
        out_coarse = self.fc_coarse(combined_features)
        out_fine = self.fc_fine(combined_features)
        return out_coarse, out_fine

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def run_experiment(seed):
    set_seed(seed)
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

    test_coarse_dataset = ImageFolder(root=os.path.join(data_dir, 'test/coarse'), transform=test_transform)
    test_fine_dataset = ImageFolder(root=os.path.join(data_dir, 'test/fine'), transform=test_transform)

    train_coarse_loader = DataLoader(train_coarse_dataset, batch_size=128, shuffle=True, num_workers=2)
    train_fine_loader = DataLoader(train_fine_dataset, batch_size=128, shuffle=True, num_workers=2)

    test_coarse_loader = DataLoader(test_coarse_dataset, batch_size=100, shuffle=False, num_workers=2)
    test_fine_loader = DataLoader(test_fine_dataset, batch_size=100, shuffle=False, num_workers=2)

    # 动态加载预训练模型
    coarse_model_path = f'best_coarse_model_seed_{seed}.pth'
    fine_model_path = f'best_fine_model_seed_{seed}.pth'

    coarse_model = ResNet9(num_classes=20).to(device)
    fine_model = ResNet9(num_classes=100).to(device)

    coarse_model.load_state_dict(torch.load(coarse_model_path, map_location=device))
    fine_model.load_state_dict(torch.load(fine_model_path, map_location=device))

    coarse_features_model = ResNet9Features(coarse_model).to(device)
    fine_features_model = ResNet9Features(fine_model).to(device)

    combined_model = CombinedModel(coarse_features_model, fine_features_model).to(device)

    for param in combined_model.coarse_features_model.parameters():
        param.requires_grad = True
    for param in combined_model.fine_features_model.parameters():
        param.requires_grad = True

    optimizer = optim.Adadelta([
        {'params': combined_model.coarse_features_model.parameters()},
        {'params': combined_model.fine_features_model.parameters()},
        {'params': combined_model.fc_coarse.parameters()},
        {'params': combined_model.fc_fine.parameters()},
    ])

    criterion_coarse = nn.CrossEntropyLoss()
    criterion_fine = nn.CrossEntropyLoss()

    num_epochs = 30
    best_coarse_accuracy = 0.0
    best_fine_accuracy = 0.0

    for epoch in range(num_epochs):
        combined_model.train()
        running_loss = 0.0
        for (coarse_data, fine_data) in tqdm(zip_longest(train_coarse_loader, train_fine_loader), total=len(train_coarse_loader), desc=f"Epoch {epoch+1}/{num_epochs}"):
            if coarse_data is None or fine_data is None:
                continue

            coarse_images, coarse_labels = coarse_data
            fine_images, fine_labels = fine_data

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

                min_batch_size = min(coarse_images.size(0), fine_images.size(0))
                coarse_images = coarse_images[:min_batch_size]
                coarse_labels = coarse_labels[:min_batch_size]
                fine_images = fine_images[:min_batch_size]
                fine_labels = fine_labels[:min_batch_size]

                coarse_images, coarse_labels = coarse_images.to(device), coarse_labels.to(device)
                fine_images, fine_labels = fine_images.to(device), fine_labels.to(device)

                out_coarse, out_fine = combined_model(coarse_images, fine_images)

                _, predicted_coarse = torch.max(out_coarse.data, 1)
                _, predicted_fine = torch.max(out_fine.data, 1)

                total_coarse += coarse_labels.size(0)
                total_fine += fine_labels.size(0)
                correct_coarse += (predicted_coarse == coarse_labels).sum().item()
                correct_fine += (predicted_fine == fine_labels).sum().item()

            coarse_accuracy = 100 * correct_coarse / total_coarse
            fine_accuracy = 100 * correct_fine / total_fine

            if coarse_accuracy > best_coarse_accuracy:
                best_coarse_accuracy = coarse_accuracy
            if fine_accuracy > best_fine_accuracy:
                best_fine_accuracy = fine_accuracy

    print(f'Best Accuracy of the model on the test images (coarse): {best_coarse_accuracy:.2f}%')
    print(f'Best Accuracy of the model on the test images (fine): {best_fine_accuracy:.2f}%')
    return best_coarse_accuracy, best_fine_accuracy

if __name__ == '__main__':
    seeds = [42, 123, 456]
    results = []

    for seed in seeds:
        print(f"Running experiment with seed {seed}")
        best_coarse_accuracy, best_fine_accuracy = run_experiment(seed)
        results.append((seed, best_coarse_accuracy, best_fine_accuracy))

    results_np = np.array(results)
    mean_coarse_acc = np.mean(results_np[:, 1])
    std_coarse_acc = np.std(results_np[:, 1])
    mean_fine_acc = np.mean(results_np[:, 2])
    std_fine_acc = np.std(results_np[:, 2])

    df = pd.DataFrame(results, columns=['Seed', 'Best Coarse Accuracy', 'Best Fine Accuracy'])
    df.loc['Mean'] = ['-', mean_coarse_acc, mean_fine_acc]
    df.loc['Std Dev'] = ['-', std_coarse_acc, std_fine_acc]

    # 显示结果
    print(df)

    # 保存结果到CSV文件
    file_exists = os.path.isfile('experiment_results_dual.csv')
    with open('experiment_results.csv', 'a') as f:
        df.to_csv(f, header=not file_exists, index=False)
