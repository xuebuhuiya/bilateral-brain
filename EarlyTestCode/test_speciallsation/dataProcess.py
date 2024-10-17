import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

__all__ = ["get_train_loader", "get_val_loader", "CIFAR100Dataset"]

class CIFAR100Dataset(Dataset):
    def __init__(self, path, transform=None, train=False):
        if train:
            sub_path = 'train'
        else:
            sub_path = 'test'
        with open(os.path.join(path, sub_path), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data[b'fine_labels'])

    def __getitem__(self, index):
        # label = self.data[b'fine_labels'][index]
        label = self.data[b'coarse_labels'][index]
        r = self.data[b'data'][index, :1024].reshape(32, 32)
        g = self.data[b'data'][index, 1024:2048].reshape(32, 32)
        b = self.data[b'data'][index, 2048:].reshape(32, 32)
        image = np.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return image, label

class CIFAR100Test(Dataset):
    def __init__(self, path, transform=None):
        with open(os.path.join(path, 'test'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data[b'data'])

    def __getitem__(self, index):
        # label = self.data[b'fine_labels'][index]
        label = self.data[b'coarse_labels'][index]
        r = self.data[b'data'][index, :1024].reshape(32, 32)
        g = self.data[b'data'][index, 1024:2048].reshape(32, 32)
        b = self.data[b'data'][index, 2048:].reshape(32, 32)
        image = np.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return image, label

def get_train_loader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cifar100_training = torchvision.datasets.CIFAR100(root='./dataset', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_val_loader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_test = torchvision.datasets.CIFAR100(root='./dataset', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def show_batch(images, labels):
    import matplotlib
    matplotlib.use('TkAgg')
    images = denormalize(images, CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    img_grid = make_grid(images, nrow=4, padding=10, normalize=True)
    plt.imshow(img_grid.permute(1, 2, 0))
    plt.title(f"Labels: {labels}")
    plt.show()

def denormalize(tensor, mean, std):
    if not torch.is_tensor(tensor):
        raise TypeError("Input should be a torch tensor.")

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)

    return tensor

def main1():
    test_loader = get_val_loader(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD, batch_size=16, num_workers=2, shuffle=False)
    for images, labels in test_loader:
        show_batch(images, labels)

if __name__ == "__main__":
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])
    train_dataset = CIFAR100Dataset(path='./dataset/cifar-100-python', transform=transform_train, train=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

    for images, labels in train_loader:
        show_batch(images, labels)
        # print(images.size(), labels)
