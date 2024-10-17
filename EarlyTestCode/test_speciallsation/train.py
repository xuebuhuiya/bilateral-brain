import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.resnet9 import ResNet9
from model.resnet import resnet9
from model.vgg11 import vgg11_bn
from model.mobilenet import mobilenetv1
from model.densenet import DenseNet

from tqdm import tqdm
import logging
import os
import io
from contextlib import redirect_stdout
from torchsummary import summary


data_dir = 'datasets_test2/CIFAR100'

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomRotation(15),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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

def main1():
    examples = enumerate(train_fine_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape, example_targets.shape)

# def main(run_number):
def main():

    # loss_functions = {
    # "CrossEntropyLoss": nn.CrossEntropyLoss(),
    # "MeanSquaredError": nn.MSELoss(),
    # "MeanAbsoluteError": nn.L1Loss()
    # }

   

    def train(model, device, train_loader, optimizer, epoch):
        # loss_func = loss_functions[loss_func_name]
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)

            # if loss_func_name in ["MeanSquaredError", "MeanAbsoluteError"]:
            #     target_one_hot = torch.zeros_like(output).scatter_(1, target.view(-1, 1), 1)
            #     loss = loss_func(output, target_one_hot)
            # else:
            #     loss = loss_func(output, target)

            loss.backward()
            optimizer.step()

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing"):
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # Sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
        return accuracy

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    log_filename_coarse = 'training_results_coarse.txt'
    log_filename_fine = 'training_results_fine.txt'
    # logging.basicConfig(filename=log_filename_coarse, level=logging.INFO, 
    #                 format='%(asctime)s - %(levelname)s - %(message)s')

    num_epochs = 45
    # 粗分类模型

    # criterion = nn.CrossEntropyLoss() 

    # coarse_model = resnet9(20).to(device)
    # coarse_model = ResNet9(num_classes=20).to(device)
    # coarse_model = DenseNet(num_classes=20).to(device)
    # logging.basicConfig(filename=log_filename_coarse, level=logging.INFO, 
    #                     format='%(asctime)s - %(levelname)s - %(message)s')
    # # coarse_optimizer = optim.Adam(coarse_model.parameters(), lr=0.0005)
    # # coarse_scheduler = optim.lr_scheduler.StepLR(coarse_optimizer, step_size=20, gamma=0.3)
    # coarse_optimizer = optim.Adadelta(coarse_model.parameters(), lr=1)
    # coarse_scheduler = optim.lr_scheduler.StepLR(coarse_optimizer, step_size=12, gamma=0.8)

    # logging.info('============================================================================')
    # logging.info(f'Model: DenseNet,total_epoch:{num_epochs}')
    # logging.info(f'Optimizer: Adadelta, Learning Rate: {coarse_optimizer.param_groups[0]["lr"]}')
    # logging.info(f'Scheduler: StepLR, Step Size: {coarse_scheduler.step_size}, Gamma: {coarse_scheduler.gamma}')

    
    # best_accuracy = 0.0

    # # 训练和评估粗分类模型
    # for epoch in range(1, num_epochs + 1):
    #     train(coarse_model, device, train_coarse_loader, coarse_optimizer, epoch)
    #     accuracy = test(coarse_model, device, test_coarse_loader)
    #     coarse_scheduler.step()

    #     if accuracy > best_accuracy:
    #         best_accuracy = accuracy
    #         torch.save(coarse_model.state_dict(), 'best_coarse_model.pth')

    # final_accuracy = test(coarse_model, device, test_coarse_loader)
    # logging.info(f'Final Accuracy for Coarse Model: {final_accuracy:.2f}%,Best_accuracy for Coarse Model: {best_accuracy:.2f}%,')




# # 训练和评估细分类模型
# 细分类模型
    # fine_model = ResNet9(num_classes=100).to(device)
    fine_model = DenseNet(num_classes=100).to(device)
    # fine_model = vgg11_bn(100).to(device)
    # f= io.StringIO()
    # with redirect_stdout(f):
    #     summary(fine_model, input_size=(3, 32, 32))
    # model_summary_str = f.getvalue()
    # fine_model = resnet9(100).to(device)
    # fine_optimizer = optim.Adam(fine_model.parameters(), lr=0.001)
    # fine_scheduler = optim.lr_scheduler.StepLR(fine_optimizer, step_size=10, gamma=0.2)
    fine_optimizer = optim.Adadelta(fine_model.parameters(), lr=1)
    fine_scheduler = optim.lr_scheduler.StepLR(fine_optimizer, step_size=12, gamma=0.8)
    # fine_optimizer = optim.AdamW(fine_model.parameters(), lr=0.0005, weight_decay=1e-4)
    # fine_scheduler = optim.lr_scheduler.StepLR(fine_optimizer, step_size=20, gamma=0.3)
    # fine_optimizer = optim.SGD(fine_model.parameters(), lr=0.001, momentum=0.9)
    # fine_scheduler = optim.lr_scheduler.StepLR(fine_optimizer, step_size=10, gamma=0.2)

    logging.basicConfig(filename=log_filename_fine, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('============================================================================')
    logging.info(f'Model: vgg11_bn,total_epoch:{num_epochs}')
    logging.info(f'Optimizer: Adam, Learning Rate: {fine_optimizer.param_groups[0]["lr"]}')
    logging.info(f'Scheduler: StepLR, Step Size: {fine_scheduler.step_size}, Gamma: {fine_scheduler.gamma}')
    # logging.info(f'Model summary:\n{model_summary_str}')

    best_accuracy = 0.0
    for epoch in range(1, num_epochs + 1):
        train(fine_model, device, train_fine_loader, fine_optimizer, epoch)
        accuracy = test(fine_model, device, test_fine_loader)
        fine_scheduler.step()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # torch.save(fine_model.state_dict(), 'best_fine_model.pth')

    final_accuracy = test(fine_model, device, test_fine_loader)
    logging.info(f'Final Accuracy for Fine Model: {final_accuracy:.2f}%,Best_accuracy for Fine Model: {best_accuracy:.2f}%,')


if __name__ == '__main__':
    main()


# if __name__ == '__main__':
#     for run_number in range(1, 6):
#         main(run_number)