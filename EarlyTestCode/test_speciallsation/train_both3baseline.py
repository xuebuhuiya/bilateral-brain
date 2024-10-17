import os
import pickle
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class CIFAR100CoarseFine(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')
        
        # 从数据文件中加载粗标签和细标签信息
        file_path = os.path.join(self.root, self.base_folder, 'train' if train else 'test')
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
        self.coarse_labels = entry['coarse_labels']
        self.fine_labels = entry['fine_labels']

    def __getitem__(self, idx):
        img, fine_target = super().__getitem__(idx)
        coarse_target = self.coarse_labels[idx]
        return img, (coarse_target, fine_target)

def main():
# 数据增强和归一化
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    
    # 使用自定义数据集类
    trainset = CIFAR100CoarseFine(root='./data', train=True, transform=transform_train)
    testset = CIFAR100CoarseFine(root='./data', train=False, transform=transform_test)

    # 数据加载器
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)



    class MultiTaskModel(nn.Module):
        def __init__(self):
            super(MultiTaskModel, self).__init__()
            base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # Remove the last fully connected layer
            self.fc_coarse = nn.Linear(base_model.fc.in_features, 20)  # Coarse classification head
            self.fc_fine = nn.Linear(base_model.fc.in_features, 100)  # Fine classification head

        def forward(self, x):
            features = self.feature_extractor(x).view(x.size(0), -1)
            coarse_preds = self.fc_coarse(features)
            fine_preds = self.fc_fine(features)
            return coarse_preds, fine_preds


    model = MultiTaskModel()
    model = model.to('cuda') if torch.cuda.is_available() else model.to('cpu')

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    def train(epoch):
        model.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loss = 0
        total = 0

        progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f'Epoch {epoch}', leave=False)
        for batch_idx, (inputs, (targets_coarse, targets_fine)) in progress_bar:
            inputs, targets_coarse, targets_fine = inputs.to(device), targets_coarse.to(device), targets_fine.to(device)
            optimizer.zero_grad()
            outputs_coarse, outputs_fine = model(inputs)
            loss_coarse = criterion(outputs_coarse, targets_coarse)
            loss_fine = criterion(outputs_fine, targets_fine)
            loss = loss_coarse + loss_fine
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total += inputs.size(0)

        print(f'Epoch: {epoch} \tTraining Loss: {train_loss / total:.6f}')

    def test():
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_loss = 0
        correct_coarse = 0
        correct_fine = 0
        total = 0

        with torch.no_grad():
            for inputs, (targets_coarse, targets_fine) in tqdm(testloader, desc='Testing', leave=False):
                inputs, targets_coarse, targets_fine = inputs.to(device), targets_coarse.to(device), targets_fine.to(device)
                outputs_coarse, outputs_fine = model(inputs)
                test_loss += criterion(outputs_coarse, targets_coarse).item() + criterion(outputs_fine, targets_fine).item()
                pred_coarse = outputs_coarse.argmax(dim=1, keepdim=True)
                pred_fine = outputs_fine.argmax(dim=1, keepdim=True)
                correct_coarse += pred_coarse.eq(targets_coarse.view_as(pred_coarse)).sum().item()
                correct_fine += pred_fine.eq(targets_fine.view_as(pred_fine)).sum().item()
                total += inputs.size(0)

        test_loss /= total
        coarse_acc = 100. * correct_coarse / total
        fine_acc = 100. * correct_fine / total
        print(f'\nTest Loss: {test_loss:.4f}, Coarse Accuracy: {coarse_acc:.2f}%, Fine Accuracy: {fine_acc:.2f}%\n')

    # 开始训练和测试
    for epoch in range(1, 21):  # 训练20个epoch
        train(epoch)
        test()

if __name__ == '__main__':
    main()