import os
import sys
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import datasets

# 加载 MNIST 数据集
mnist_train = datasets.MNIST(root='mnist_data', train=True, download=True)
mnist_test = datasets.MNIST(root='mnist_data', train=False, download=True)
mnist_data = np.concatenate((mnist_train.data.numpy(), mnist_test.data.numpy()), axis=0)
mnist_labels = np.concatenate((mnist_train.targets.numpy(), mnist_test.targets.numpy()), axis=0)

# 按数字标签组织图像
digit_images = {i: [] for i in range(10)}
for img, label in zip(mnist_data, mnist_labels):
    digit_images[label].append(img)

# 创建合成图像的函数
def create_composite_image(coarse_digit_label, fine_digit_label, digit_images, image_size=128):
    # 从 MNIST 数据集中随机选取一个大数字图像
    coarse_digit_img = random.choice(digit_images[coarse_digit_label])
    coarse_digit_img = Image.fromarray(coarse_digit_img)
    # 将大数字图像放大到指定尺寸
    coarse_digit_img = coarse_digit_img.resize((image_size, image_size), resample=Image.BILINEAR)
    # 转为二值化掩码
    coarse_digit_img = coarse_digit_img.point(lambda p: p > 50 and 255)
    mask = np.array(coarse_digit_img)
    # 创建一个空白图像
    composite_img = Image.new('L', (image_size, image_size), color=0)
    # 获取小数字的图像
    small_digit_size = 14  # 小数字的尺寸，可以根据需要调整
    fine_digit_images = random.choices(digit_images[fine_digit_label], k=1000)
    fine_digit_images = [
        Image.fromarray(img).resize((small_digit_size, small_digit_size), resample=Image.BILINEAR)
        for img in fine_digit_images
    ]
    # 获取掩码中非零像素的位置
    positions = np.argwhere(mask > 0)
    # 随机选择部分位置，避免过于密集
    positions = [tuple(pos) for pos in positions]
    random.shuffle(positions)
    positions = positions[:len(fine_digit_images)]
    # 在掩码区域内粘贴小数字
    for pos, img in zip(positions, fine_digit_images):
        x = pos[1] - small_digit_size // 2
        y = pos[0] - small_digit_size // 2
        if 0 <= x < image_size - small_digit_size and 0 <= y < image_size - small_digit_size:
            composite_img.paste(img, (x, y))
    return composite_img

# 创建目录结构
def create_directories(base_path):
    for split in ['train', 'test']:
        for category in ['coarse', 'fine']:
            for digit in range(10):
                path = os.path.join(base_path, split, category, str(digit))
                os.makedirs(path, exist_ok=True)

create_directories('sb_mnist3')

# 生成并保存图像
def generate_images(num_images, split):
    for _ in tqdm(range(num_images), desc=f"Generating {split} images"):
        coarse_digit = random.randint(0, 9)
        fine_digit = random.randint(0, 9)
        img = create_composite_image(coarse_digit, fine_digit, digit_images)
        img_name = f"{coarse_digit}_{fine_digit}_{random.randint(0, 1e6)}.png"
        coarse_path = os.path.join('sb_mnist3', split, 'coarse', str(coarse_digit), img_name)
        fine_path = os.path.join('sb_mnist3', split, 'fine', str(fine_digit), img_name)
        img.save(coarse_path)
        img.save(fine_path)

# 开始生成图像
generate_images(120000, 'train')
generate_images(20000, 'test')
