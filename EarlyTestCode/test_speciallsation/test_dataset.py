import pickle
import numpy as np

def load_cifar100(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

# 加载元数据
meta_path = 'dataset/cifar-100-python/meta'
meta = load_cifar100(meta_path)
fine_label_names = meta[b'fine_label_names']
coarse_label_names = meta[b'coarse_label_names']

# 加载训练和测试数据
train_path = 'dataset/cifar-100-python/train'
test_path = 'dataset/cifar-100-python/test'
train_data = load_cifar100(train_path)
test_data = load_cifar100(test_path)

# 检查数据
print(f"训练集图像数量: {len(train_data[b'data'])}")
print(f"测试集图像数量: {len(test_data[b'data'])}")
print(f"细分类别名称: {fine_label_names}")
print(f"粗分类别名称: {coarse_label_names}")

# 数据处理
train_images = train_data[b'data'].reshape((len(train_data[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)
test_images = test_data[b'data'].reshape((len(test_data[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)
print(f"训练集图像形状: {train_images.shape}")
print(f"测试集图像形状: {test_images.shape}")
