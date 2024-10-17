# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
#Define normalization 
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    #Load dataset
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                    transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                    transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64, shuffle=True,num_workers=2, persistent_workers=False)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=64, shuffle=True,num_workers=2, persistent_workers=False)

    #Build the model we defined above
    model = Lenet().to(device)

    #Define the optimizer for model training
    optimizer = optim.Adadelta(model.parameters(), lr=1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    train_losses = []
    test_losses = []
    test_accuracies = []

    # for epoch in range(1, 6):
    #     model.train()
    #     for batch_idx, (data, target) in enumerate(train_loader):
    #         optimizer.zero_grad()
    #         output = model(data)
    #         loss = F.nll_loss(output, target)
    #         loss.backward()
    #         optimizer.step()
    #         if batch_idx % 10 == 0:
    #             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #                 epoch, batch_idx * len(data), len(train_loader.dataset),
    #                 100. * batch_idx / len(train_loader), loss.item()))
    #     scheduler.step()

    #     model.eval()
    #     test_loss = 0
    #     correct = 0
    #     with torch.no_grad():
    #         for data, target in test_loader:
    #             output = model(data)
    #             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
    #             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    #             correct += pred.eq(target.view_as(pred)).sum().item()

    #     test_loss /= len(test_loader.dataset)

    #     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #         test_loss, correct, len(test_loader.dataset),
    #         100. * correct / len(test_loader.dataset)))
    # accoding assignment instruction change the model name to target_model.pth



    for epoch in range(1, 6):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct_train += pred.eq(target.view_as(pred)).sum().item()
            total_train += target.size(0)
            
        scheduler.step()
        
        train_loss /= len(train_loader.dataset)
        train_accuracy = 100. * correct_train / total_train
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0
        correct = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())


        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * correct / len(test_loader.dataset)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f'\nEpoch: {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_accuracy:.2f}%)\n')

        cm = confusion_matrix(all_targets, [p[0] for p in all_preds])
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for Epoch {epoch}')
        # plt.show()

    torch.save(model.state_dict(), "lenet_model.pth")

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 6), train_losses, label='Train Loss')
    plt.plot(range(1, 6), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.show()

    # 绘制准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 6), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.show()

if __name__ == '__main__':
    main()