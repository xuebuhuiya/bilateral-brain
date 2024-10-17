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
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

class AttentionFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionFusion, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, coarse_features, fine_features):
        combined_features = torch.cat((coarse_features, fine_features), dim=1)
        attention_weights = torch.sigmoid(self.fc2(torch.relu(self.fc1(combined_features))))
        fused_features = combined_features * attention_weights
        return fused_features

class CombinedModel(nn.Module):
    def __init__(self, coarse_features_model, fine_features_model):
        super(CombinedModel, self).__init__()
        self.coarse_features_model = coarse_features_model
        self.fine_features_model = fine_features_model
        self.attention_fusion = AttentionFusion(8192*2, 1024)
        self.fc1 = nn.Linear(8192*2, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.5)
        self.fc_coarse = nn.Linear(1024, 20)
        self.fc_fine = nn.Linear(1024, 100)

    def forward(self, x_coarse, x_fine):
        coarse_features = self.coarse_features_model(x_coarse)
        fine_features = self.fine_features_model(x_fine)
        fused_features = self.attention_fusion(coarse_features, fine_features)
        x = torch.relu(self.fc1(fused_features))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        out_coarse = self.fc_coarse(x)
        out_fine = self.fc_fine(x)
        return out_coarse, out_fine

def load_data(data_dir, train_transform, test_transform):
    train_coarse_dataset = ImageFolder(root=os.path.join(data_dir, 'train/coarse'), transform=train_transform)
    train_fine_dataset = ImageFolder(root=os.path.join(data_dir, 'train/fine'), transform=train_transform)

    test_coarse_dataset = ImageFolder(root=os.path.join(data_dir, 'test/coarse'), transform=test_transform)
    test_fine_dataset = ImageFolder(root=os.path.join(data_dir, 'test/fine'), transform=test_transform)

    train_coarse_loader = DataLoader(train_coarse_dataset, batch_size=128, shuffle=True, num_workers=2)
    train_fine_loader = DataLoader(train_fine_dataset, batch_size=128, shuffle=True, num_workers=2)

    test_coarse_loader = DataLoader(test_coarse_dataset, batch_size=100, shuffle=False, num_workers=2)
    test_fine_loader = DataLoader(test_fine_dataset, batch_size=100, shuffle=False, num_workers=2)

    return train_coarse_loader, train_fine_loader, test_coarse_loader, test_fine_loader

def main():
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
    train_coarse_loader, train_fine_loader, test_coarse_loader, test_fine_loader = load_data(data_dir, train_transform, test_transform)

    coarse_model = ResNet9(num_classes=20).to(device)
    fine_model = ResNet9(num_classes=100).to(device)

    coarse_model.load_state_dict(torch.load('best_coarse_model.pth', map_location=device))
    fine_model.load_state_dict(torch.load('best_fine_model.pth', map_location=device))

    coarse_features_model = ResNet9Features(coarse_model).to(device)
    fine_features_model = ResNet9Features(fine_model).to(device)
    
    combined_model = CombinedModel(coarse_features_model, fine_features_model).to(device)
    
    for param in combined_model.coarse_features_model.parameters():
        param.requires_grad = False
    for param in combined_model.fine_features_model.parameters():
        param.requires_grad = False

    optimizer = optim.Adam([
        {'params': combined_model.fc1.parameters()},
        {'params': combined_model.bn1.parameters()},
        {'params': combined_model.dropout1.parameters()},
        {'params': combined_model.fc2.parameters()},
        {'params': combined_model.bn2.parameters()},
        {'params': combined_model.dropout2.parameters()},
        {'params': combined_model.fc_coarse.parameters()},
        {'params': combined_model.fc_fine.parameters()},
    ], lr=1e-3)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    criterion_coarse = nn.CrossEntropyLoss()
    criterion_fine = nn.CrossEntropyLoss()

    num_epochs = 30
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

        scheduler.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_coarse_loader):.4f}')

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

        print(f'Accuracy of the model on the test images (coarse): {100 * correct_coarse / total_coarse:.2f}%')
        print(f'Accuracy of the model on the test images (fine): {100 * correct_fine / total_fine:.2f}%')

if __name__ == '__main__':
    main()
