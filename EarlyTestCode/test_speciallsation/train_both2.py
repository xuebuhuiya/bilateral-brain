import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import os
from PIL import Image
from tqdm import tqdm
import json
# 数据准备
data_dir = 'datasets_test2/CIFAR100'
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(32, padding=4),
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

# model = MultiTaskModel()
# criterion_coarse = nn.CrossEntropyLoss()
# criterion_fine = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# 训练模型
def train(model, train_coarse_loader, train_fine_loader, optimizer, criterion_coarse, criterion_fine, device):
    model.train()
    for (data_coarse, target_coarse), (data_fine, target_fine) in tqdm(zip(train_coarse_loader, train_fine_loader), total=min(len(train_coarse_loader), len(train_fine_loader)), desc="Training"):
        data_coarse, target_coarse = data_coarse.to(device), target_coarse.to(device)
        data_fine, target_fine = data_fine.to(device), target_fine.to(device)
        optimizer.zero_grad()
        coarse_output, fine_output = model(data_coarse)
        loss_coarse = criterion_coarse(coarse_output, target_coarse)
        fine_output, fine_output = model(data_fine)
        loss_fine = criterion_fine(fine_output, target_fine)
        loss = loss_coarse + loss_fine #one in left one in right
        loss.backward()
        optimizer.step()

# 评估模型
def evaluate(model, test_coarse_loader, test_fine_loader, device):
    model.eval()
    correct_coarse = 0
    correct_fine = 0
    total_coarse = 0
    total_fine = 0
    with torch.no_grad():
        for data_coarse, target_coarse in tqdm(test_coarse_loader, desc="Evaluating Coarse"):
            data_coarse, target_coarse = data_coarse.to(device), target_coarse.to(device)
            coarse_output, _ = model(data_coarse)
            _, predicted_coarse = torch.max(coarse_output, 1)
            total_coarse += target_coarse.size(0)
            correct_coarse += (predicted_coarse == target_coarse).sum().item()
        
        for data_fine, target_fine in tqdm(test_fine_loader, desc="Evaluating Fine"):
            data_fine, target_fine = data_fine.to(device), target_fine.to(device)
            _, fine_output = model(data_fine)
            _, predicted_fine = torch.max(fine_output, 1)
            total_fine += target_fine.size(0)
            correct_fine += (predicted_fine == target_fine).sum().item()

    coarse_accuracy = 100 * correct_coarse / total_coarse
    fine_accuracy = 100 * correct_fine / total_fine
    print(f'Coarse Accuracy: {coarse_accuracy:.2f}%')
    print(f'Fine Accuracy: {fine_accuracy:.2f}%')
    return coarse_accuracy, fine_accuracy

# 预测单张图片
def predict(image_path, model, device):
    model.eval()
    image = Image.open(image_path)
    image = test_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        coarse_output, fine_output = model(image)
        probs_coarse = F.softmax(coarse_output, dim=1)
        probs_fine = F.softmax(fine_output, dim=1)
        top5_coarse_probs, top5_coarse_classes = probs_coarse.topk(5, dim=1)
        top5_fine_probs, top5_fine_classes = probs_fine.topk(5, dim=1)
        return top5_coarse_probs, top5_coarse_classes, top5_fine_probs, top5_fine_classes

def predict_folder(folder_path, model, device):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(folder_path, filename)
            top5_coarse_probs, top5_coarse_classes, top5_fine_probs, top5_fine_classes = predict(image_path, model, device)
            results.append({
                "filename": filename,
                "top5_coarse_classes": top5_coarse_classes.cpu().numpy().tolist(),
                "top5_coarse_probs": top5_coarse_probs.cpu().numpy().tolist(),
                "top5_fine_classes": top5_fine_classes.cpu().numpy().tolist(),
                "top5_fine_probs": top5_fine_probs.cpu().numpy().tolist()
            })
            # 打印结果
            print(f'Image: {filename}')
            print(f'Top-5 Coarse Classes: {top5_coarse_classes}')
            print(f'Top-5 Coarse Probabilities: {top5_coarse_probs}')
            print(f'Top-5 Fine Classes: {top5_fine_classes}')
            print(f'Top-5 Fine Probabilities: {top5_fine_probs}')
            print('-' * 50)
    return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # 训练和评估
    model = MultiTaskModel().to(device)
    # criterion_coarse = nn.CrossEntropyLoss()
    # criterion_fine = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # for epoch in range(20):
    #     train(model, train_coarse_loader, train_fine_loader, optimizer, criterion_coarse, criterion_fine, device)
    #     coarse_accuracy, fine_accuracy = evaluate(model, test_coarse_loader, test_fine_loader, device)
    #     scheduler.step()

    # torch.save(model.state_dict(), 'multitask_model.pth')


    
    model.load_state_dict(torch.load('multitask_model.pth'))
    model.to(device)
    # evaluate(model, test_coarse_loader, test_fine_loader, device)
    
    # # 示例图片路径
    # image_path = 'datasets_test2/CIFAR100/test/coarse/14/adam_s_001319.png'
    # # 预测新图片
    # top5_coarse_probs, top5_coarse_classes, top5_fine_probs, top5_fine_classes = predict(image_path, model, device)

    # # 打印结果
    # print(f'Top-5 Coarse Classes: {top5_coarse_classes}')
    # print(f'Top-5 Coarse Probabilities: {top5_coarse_probs}')
    # print(f'Top-5 Fine Classes: {top5_fine_classes}')
    # print(f'Top-5 Fine Probabilities: {top5_fine_probs}')

    folder_path = 'datasets_test2/CIFAR100/test/coarse/18/'
    
    # # 预测文件夹中的所有图片
    results = predict_folder(folder_path, model, device)
    
    # # 结果可以进一步处理或保存
    # # 保存到一个 JSON 文件
    
    with open('predictions.json', 'w') as f:
        json.dump(results, f, indent=4)
