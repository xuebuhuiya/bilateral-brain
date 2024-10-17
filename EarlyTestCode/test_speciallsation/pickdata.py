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
target_dir = 'datasets_test/CIFAR100'  # the output images go here

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

def save_images(data, fine_labels_indices, coarse_labels_indices, split):
    split_dir = osp.join(target_dir, split)
    coarse_split_dir = osp.join(split_dir, 'coarse')
    fine_split_dir = osp.join(split_dir, 'fine')

    for i, (img, fine_label_idx, coarse_label_idx) in enumerate(tqdm(zip(data, fine_labels_indices, coarse_labels_indices))):
        coarse_label_name = coarse_labels[coarse_label_idx]
        fine_label_name = fine_labels[fine_label_idx]

        img = img.reshape(3, 32, 32).transpose(1, 2, 0)  # Reshape and transpose the image to (32, 32, 3)
        img_name = f'{split}_{i}.png'
        
        # Save coarse images
        coarse_img_dir = osp.join(coarse_split_dir, coarse_label_name)
        os.makedirs(coarse_img_dir, exist_ok=True)
        coarse_img_path = osp.join(coarse_img_dir, img_name)
        cv2.imwrite(coarse_img_path, img)
        
        # Save fine images
        fine_img_dir = osp.join(fine_split_dir, fine_label_name)
        os.makedirs(fine_img_dir, exist_ok=True)
        fine_img_path = osp.join(fine_img_dir, img_name)
        cv2.imwrite(fine_img_path, img)

# Load training data
train_data = unpickle(osp.join(raw_data_path, 'train'))
train_images = train_data[b'data']
train_fine_labels = train_data[b'fine_labels']
train_coarse_labels = train_data[b'coarse_labels']

# Ensure data is of the correct type
train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

# Save training images
save_images(train_images, train_fine_labels, train_coarse_labels, 'train')

# Load test data
test_data = unpickle(osp.join(raw_data_path, 'test'))
test_images = test_data[b'data']
test_fine_labels = test_data[b'fine_labels']
test_coarse_labels = test_data[b'coarse_labels']

# Ensure data is of the correct type
test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

# Save test images
save_images(test_images, test_fine_labels, test_coarse_labels, 'test')
