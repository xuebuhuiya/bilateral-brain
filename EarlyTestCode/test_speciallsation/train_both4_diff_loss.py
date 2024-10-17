import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import os
import pickle
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        centers_batch = self.centers.index_select(0, labels)
        loss = torch.sum((features - centers_batch) ** 2) / 2.0 / features.size(0)
        return loss

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class MultiTaskModel(nn.Module):
        def __init__(self):
            super(MultiTaskModel, self).__init__()
            base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            self.dropout = nn.Dropout(0.5)
            self.fc_coarse = nn.Linear(base_model.fc.in_features, 20)
            self.fc_fine = nn.Linear(base_model.fc.in_features, 100)

        def forward(self, x):
            features = self.feature_extractor(x).view(x.size(0), -1)
            features = self.dropout(features)
            coarse_preds = self.fc_coarse(features)
            fine_preds = self.fc_fine(features)
            return coarse_preds, fine_preds, features

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


def get_triplets(features, labels):
        """
        Generate triplets for training from batch data.
        :param features: Tensor, containing the output of the feature extractor for the batch.
        :param labels: Tensor, containing labels of the batch.
        :return: tuple of Tensors (anchors, positives, negatives)
        """
        labels = labels.cpu().data.numpy()
        unique_labels = np.unique(labels)
        anchor_list = []
        positive_list = []
        negative_list = []

        for label in unique_labels:
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue  # Skip if less than 2 examples of the same class
            non_label_indices = np.where(np.logical_not(label_mask))[0]

            # Random choice of anchor and positive
            anchor_positives = np.random.choice(label_indices, size=2, replace=False)
            anchor_idx = anchor_positives[0]
            positive_idx = anchor_positives[1]

            # Random choice of negative
            negative_idx = np.random.choice(non_label_indices)

            anchor_list.append(features[anchor_idx])
            positive_list.append(features[positive_idx])
            negative_list.append(features[negative_idx])

        # Stack lists to form new tensors
        return torch.stack(anchor_list), torch.stack(positive_list), torch.stack(negative_list)


def adjust_loss_weights(epoch, base_weight=0.01, increase_factor=0.02, performance_metric=None):
        """
        Adjust the weights of the auxiliary losses based on some performance metrics.
        """
        if performance_metric is not None:
            if performance_metric < 0.6:  # Assume some threshold for performance
                return base_weight + increase_factor
            else:
                return max(base_weight - increase_factor, 0.01)
        return base_weight

# def adjust_loss_weights(epoch, base_weight=0.01, increase_factor=0.02, performance_metric=None):
#     """
#     Adjust the weights of the auxiliary losses based on some performance metrics.
#     """
#     if performance_metric is not None:
#         if performance_metric < 0.6:  # Assume some threshold for performance
#             return base_weight + increase_factor
#         else:
#             return max(base_weight - increase_factor, 0.01)
#     # 在训练的前10个epoch内保持权重不变
#     if epoch < 10:
#         return base_weight
#     return base_weight + (epoch - 10) * increase_factor


def train(epoch, model, train_loader, optimizer, criterion, device, center_loss, triplet_loss, coarse_weight, fine_weight,center_loss_weight, triplet_loss_weight):
        model.train()
        total_loss = 0
        total_loss_coarse = 0
        total_loss_fine = 0
        total_center_loss = 0
        total_triplet_loss = 0

        performance_metric = None  # Some logic to obtain performance metric, e.g., validation accuracy

        center_loss_weight = adjust_loss_weights(epoch, performance_metric=performance_metric)
        triplet_loss_weight = adjust_loss_weights(epoch, performance_metric=performance_metric, base_weight=0.005)


        for batch_idx, (inputs, (targets_coarse, targets_fine)) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}', leave=False):
            inputs, targets_coarse, targets_fine = inputs.to(device), targets_coarse.to(device), targets_fine.to(device)
            optimizer.zero_grad()
            outputs_coarse, outputs_fine, features = model(inputs)
            # 基础分类损失
            loss_coarse = criterion(outputs_coarse, targets_coarse) * coarse_weight
            loss_fine = criterion(outputs_fine, targets_fine) * fine_weight

            # 特征级别的辅助损失
            center_loss_val = center_loss(features, targets_coarse)  # 假设中心损失帮助粗分类

            # Retrieve triplets
            anchors, positives, negatives = get_triplets(features.detach(), targets_fine)
            triplet_loss_val = triplet_loss(anchors, positives, negatives)

            # Combine losses
            # 将辅助损失整合到主损失中
            total_coarse_loss = loss_coarse + center_loss_weight * center_loss_val
            total_fine_loss = loss_fine + triplet_loss_weight * triplet_loss_val
            loss = total_coarse_loss + total_fine_loss
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_coarse += loss_coarse.item()
            total_loss_fine += loss_fine.item()
            total_center_loss += center_loss_val.item()
            total_triplet_loss += triplet_loss_val.item()
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}: loss_coarse={loss_coarse.item():.4f}, loss_fine={loss_fine.item():.4f}, center_loss_val={center_loss_val.item():.4f}, triplet_loss_val={triplet_loss_val.item():.4f}')
        
        
        avg_loss = total_loss / len(train_loader)
        avg_loss_coarse = total_loss_coarse / len(train_loader)
        avg_loss_fine = total_loss_fine / len(train_loader)
        avg_center_loss = total_center_loss / len(train_loader)
        avg_triplet_loss = total_triplet_loss / len(train_loader)
        
        
        print(f'Epoch: {epoch} \tTraining Loss: {avg_loss:.6f} \tLoss Coarse: {avg_loss_coarse:.6f} \tLoss Fine: {avg_loss_fine:.6f} \tCenter Loss: {avg_center_loss:.6f} \tTriplet Loss: {avg_triplet_loss:.6f}')


def test(model, test_loader, criterion, device):
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_loss = 0
        test_loss_coarse = 0
        test_loss_fine = 0
        correct_coarse = 0
        correct_fine = 0
        total = 0

        with torch.no_grad():
            for inputs, (targets_coarse, targets_fine) in tqdm(test_loader, desc='Testing', leave=False):
                inputs, targets_coarse, targets_fine = inputs.to(device), targets_coarse.to(device), targets_fine.to(device)
                outputs_coarse, outputs_fine, _ = model(inputs)  # Assume the model's forward returns features as third output, which we ignore here
                test_loss_coarse = criterion(outputs_coarse, targets_coarse)
                test_loss_fine = criterion(outputs_fine, targets_fine)
                loss = test_loss_coarse + test_loss_fine
                test_loss += loss.item() * inputs.size(0)
                test_loss_coarse += test_loss_coarse.item() * inputs.size(0)
                test_loss_fine += test_loss_fine.item() * inputs.size(0)
                pred_coarse = outputs_coarse.argmax(dim=1, keepdim=True)
                pred_fine = outputs_fine.argmax(dim=1, keepdim=True)
                correct_coarse += pred_coarse.eq(targets_coarse.view_as(pred_coarse)).sum().item()
                correct_fine += pred_fine.eq(targets_fine.view_as(pred_fine)).sum().item()
                total += inputs.size(0)

        avg_test_loss = test_loss / total
        avg_test_loss_coarse = test_loss_coarse / total
        avg_test_loss_fine = test_loss_fine / total
        coarse_accuracy = 100. * correct_coarse / total
        fine_accuracy = 100. * correct_fine / total
        print(f'\nTest set: Average loss: {avg_test_loss:.4f}, Coarse Loss: {avg_test_loss_coarse:.4f}, Fine Loss: {avg_test_loss_fine:.4f}, Coarse Accuracy: {coarse_accuracy:.2f}%, Fine Accuracy: {fine_accuracy:.2f}%\n')
        return avg_test_loss


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

    # model = MultiTaskModel()
    # model = model.to('cuda') if torch.cuda.is_available() else model.to('cpu')

    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.15, min_lr=5*1e-7, verbose=True)
    criterion = nn.CrossEntropyLoss()

    # 中心损失和三元组损失的实例
    feat_dim = model.fc_fine.in_features  # 假设最后全连接层的输入特征维度是模型特征维度
    center_loss = CenterLoss(100, feat_dim, device).to(device)
    triplet_loss = TripletLoss(margin=1.0).to(device)

    # 指定辅助损失的权重
    center_loss_weight = 0.002
    triplet_loss_weight = 0.2
    coarse_weight = 1.0
    fine_weight = 1.0

    # 初始化早停机制
    early_stopping = EarlyStopping(patience=10, verbose=True)

    # 开始训练和测试
    for epoch in range(1, 51):
        train(epoch, model, trainloader, optimizer, criterion, device, center_loss, triplet_loss, coarse_weight, fine_weight, center_loss_weight, triplet_loss_weight)
        val_loss = test(model, testloader, criterion, device)
        scheduler.step(val_loss)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # 加载早停前保存的最佳模型
        model.load_state_dict(torch.load('checkpoint.pt'))

if __name__ == '__main__':
    main()