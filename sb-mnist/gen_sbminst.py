import torch
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image, ImageEnhance
import random

# 下载并导入MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载MNIST训练和测试数据
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 函数：根据大图像中的非空区域，用小图像替换并生成sb-MNIST图片
def create_sb_mnist_image(big_digit, small_digit, image_size=100, small_image_size=8, sparsity=0.7):
    # 获取大图像并转换为PIL Image对象
    big_digit_array = train_data.data[big_digit].numpy()  # 大数字（全局结构）
    big_digit_image = Image.fromarray(big_digit_array)  # 转换为PIL Image
    
    # 将大图像调整为所需的尺寸（如100x100）
    big_digit_image = big_digit_image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    
    # 获取小图像并缩放
    small_digit_array = train_data.data[small_digit].numpy()  # 小数字（局部结构）
    small_digit_image = Image.fromarray(small_digit_array).resize((small_image_size, small_image_size), Image.Resampling.LANCZOS)  # 小图像缩放为8x8
    
    # 增强小数字对比度，提升可见性
    enhancer = ImageEnhance.Contrast(small_digit_image)
    small_digit_image = enhancer.enhance(2.0)  # 增强对比度
    
    # 创建一个空白图片，用于存储结果
    sb_image = Image.new('L', (image_size, image_size))

    # 将大图像转换回numpy数组，用于非空像素检测
    big_digit_array_resized = np.array(big_digit_image)  # 将大图像再次转换为数组
    big_digit_array_resized = (big_digit_array_resized > 0).astype(np.uint8)  # 转换为二值图像
    
    # 遍历大图像中的像素，使用小图像替换稀疏的非空像素
    for i in range(0, image_size, small_image_size):  # 以8x8的步长遍历
        for j in range(0, image_size, small_image_size):
            if big_digit_array_resized[i, j] != 0 and random.random() < sparsity:  # 使用稀疏性控制小数字数量
                sb_image.paste(small_digit_image, (i, j))  # 在这个位置用小图像进行替换
    
    return sb_image

# 设置保存路径
def save_image(image, label, folder, idx, fine_or_coarse):
    os.makedirs(os.path.join(folder, fine_or_coarse, str(label)), exist_ok=True)
    image.save(os.path.join(folder, fine_or_coarse, str(label), f"{idx}.png"))

# 生成并保存sb-MNIST数据集
def generate_sb_mnist_dataset(data, num_images, output_folder):
    fine_labels = []
    coarse_labels = []
    
    for i in range(num_images):
        # 随机选择一个大数字和一个小数字
        big_digit = random.randint(0, 9)
        small_digit = random.randint(0, 9)
        
        # 创建sb-MNIST图片
        sb_image = create_sb_mnist_image(big_digit, small_digit)
        
        # 保存图片分别到 coarse 和 fine 文件夹
        save_image(sb_image, big_digit, output_folder, i, 'coarse')
        save_image(sb_image, small_digit, output_folder, i, 'fine')
        
        # 存储标签
        coarse_labels.append(big_digit)
        fine_labels.append(small_digit)
    
    # 返回标签数组
    return coarse_labels, fine_labels

# 生成训练集和测试集
train_num = 120000
test_num = 20000
output_folder = './sb_mnist_dataset'

# 生成训练集
train_coarse_labels, train_fine_labels = generate_sb_mnist_dataset(train_data, train_num, output_folder)

# 生成测试集
test_coarse_labels, test_fine_labels = generate_sb_mnist_dataset(test_data, test_num, output_folder)

# 保存标签
np.save(os.path.join(output_folder, 'train_coarse_labels.npy'), train_coarse_labels)
np.save(os.path.join(output_folder, 'train_fine_labels.npy'), train_fine_labels)
np.save(os.path.join(output_folder, 'test_coarse_labels.npy'), test_coarse_labels)
np.save(os.path.join(output_folder, 'test_fine_labels.npy'), test_fine_labels)

print("数据集生成完成。")
