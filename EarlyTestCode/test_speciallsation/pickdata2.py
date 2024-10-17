import os
import cv2
import numpy as np
import os.path as osp
from tqdm import tqdm

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

metadata_path = 'dataset/cifar-100-python/meta'  # source cifar 'meta' file
raw_data_path = 'dataset/cifar-100-python/'  # source cifar files here (must have a trailing slash)
target_dir = 'datasets_test2/CIFAR100'  # the output images go here

# Load metadata
meta = unpickle(metadata_path)
fine_labels = [label.decode('utf-8') for label in meta[b'fine_label_names']]
coarse_labels = [label.decode('utf-8') for label in meta[b'coarse_label_names']]

# Create directories for train and test sets
for split in ['train', 'test']:
    split_dir = osp.join(target_dir, split)
    coarse_split_dir = osp.join(split_dir, 'coarse')
    fine_split_dir = osp.join(split_dir, 'fine')
    os.makedirs(coarse_split_dir, exist_ok=True)
    os.makedirs(fine_split_dir, exist_ok=True)

def save_images(data_dic, fine_labels_indices, coarse_labels_indices, filenames_dic, target_dir):
    for split in ['train', 'test']:
        split_dir = osp.join(target_dir, split)
        coarse_split_dir = osp.join(split_dir, 'coarse')
        fine_split_dir = osp.join(split_dir, 'fine')

        os.makedirs(coarse_split_dir, exist_ok=True)
        os.makedirs(fine_split_dir, exist_ok=True)

        coarse_label_dic = coarse_labels_indices[split]
        fine_label_dic = fine_labels_indices[split]

        print('----> writing images for {}'.format(split))
        for i in tqdm(range(data_dic[split].shape[0])):
            # 将图像数据从一维数组转换为32x32的三通道图像（RGB）
            r = data_dic[split][i][:1024].reshape(32, 32)
            g = data_dic[split][i][1024:2048].reshape(32, 32)
            b = data_dic[split][i][2048:].reshape(32, 32)
            img = np.stack([b, g, r]).transpose((1, 2, 0))

            # 确保图像数据的范围在0到255之间，并且类型为uint8
            img = np.clip(img, 0, 255).astype(np.uint8)

            # 保存图像
            for label_dic, label_type in zip([coarse_label_dic, fine_label_dic], ['coarse', 'fine']):
                label_name = label_dic[i]
                out_dir = osp.join(split_dir, label_type, str(label_name))
                os.makedirs(out_dir, exist_ok=True)
                img_name = str(filenames_dic[split][i], 'utf-8')
                img_path = osp.join(out_dir, img_name)
                cv2.imwrite(img_path, img)

# Load training data
train_data = unpickle(osp.join(raw_data_path, 'train'))
train_images = train_data[b'data']
train_fine_labels = train_data[b'fine_labels']
train_coarse_labels = train_data[b'coarse_labels']
train_filenames = train_data[b'filenames']

# Load test data
test_data = unpickle(osp.join(raw_data_path, 'test'))
test_images = test_data[b'data']
test_fine_labels = test_data[b'fine_labels']
test_coarse_labels = test_data[b'coarse_labels']
test_filenames = test_data[b'filenames']

# Ensure data is of the correct type
data_dic = {
    'train': train_images,
    'test': test_images
}
fine_labels_indices = {
    'train': train_fine_labels,
    'test': test_fine_labels
}
coarse_labels_indices = {
    'train': train_coarse_labels,
    'test': test_coarse_labels
}
filenames_dic = {
    'train': train_filenames,
    'test': test_filenames
}

# Save images
save_images(data_dic, fine_labels_indices, coarse_labels_indices, filenames_dic, target_dir)
