import os
import random
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import ImageFont

font = ImageFont.load_default()
# 步骤1：下载 MNIST 数据集
mnist_train = datasets.MNIST(root='mnist_data', train=True, download=True)
mnist_test = datasets.MNIST(root='mnist_data', train=False, download=True)

# 合并训练和测试数据集
mnist_data = mnist_train.data.numpy()
mnist_labels = mnist_train.targets.numpy()
mnist_data = np.concatenate((mnist_data, mnist_test.data.numpy()), axis=0)
mnist_labels = np.concatenate((mnist_labels, mnist_test.targets.numpy()), axis=0)

# 按数字标签组织 MNIST 图像
digit_images = {i: [] for i in range(10)}
for img, label in zip(mnist_data, mnist_labels):
    digit_images[label].append(img)

# 修改后的 create_composite_image 函数
def create_composite_image(coarse_digit, fine_digit, digit_images, image_size=64):
    # 创建一个空白图像
    composite_img = Image.new('L', (image_size, image_size), color=0)

    # 创建粗数字的掩码
    font_size = 56
    font = ImageFont.truetype("arial.ttf", font_size)
    mask_img = Image.new('L', (image_size, image_size), color=0)
    draw = ImageDraw.Draw(mask_img)
    # 使用 textbbox 获取文本尺寸
    bbox = draw.textbbox((0, 0), str(coarse_digit), font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((image_size - w) / 2, (image_size - h) / 2),
        str(coarse_digit),
        fill=255,
        font=font
    )
    mask = np.array(mask_img)

    # 获取小数字的图像
    small_digit_size = 8
    fine_digit_images = random.choices(digit_images[fine_digit], k=500)
    fine_digit_images = [
        Image.fromarray(img).resize((small_digit_size, small_digit_size))
        for img in fine_digit_images
    ]

    # 生成要粘贴小数字的位置
    positions = []
    for y in range(0, image_size, small_digit_size):
        for x in range(0, image_size, small_digit_size):
            # 检查网格中心点是否在掩码中
            center_y = y + small_digit_size // 2
            center_x = x + small_digit_size // 2
            if center_y >= image_size or center_x >= image_size:
                continue
            if mask[center_y, center_x] > 0:
                positions.append((x, y))
    
    random.shuffle(positions)

    # 在掩码内粘贴小数字
    for pos in positions:
        if len(fine_digit_images) == 0:
            break
        img = fine_digit_images.pop()
        x, y = pos
        composite_img.paste(img, (x, y))

    return composite_img

# 步骤3：创建目录
def create_directories(base_path):
    for split in ['train', 'test']:
        for category in ['coarse', 'fine']:
            for digit in range(10):
                path = os.path.join(base_path, split, category, str(digit))
                os.makedirs(path, exist_ok=True)

create_directories('sb_mnist2')

# 步骤4：生成并保存图像
def generate_images(num_images, split):
    for _ in tqdm(range(num_images), desc=f"Generating {split} images"):
        coarse_digit = random.randint(0, 9)
        fine_digit = random.randint(0, 9)
        img = create_composite_image(coarse_digit, fine_digit, digit_images)
        img_name = f"{coarse_digit}_{fine_digit}_{random.randint(0, 1e6)}.png"

        # 保存到 coarse 目录
        coarse_path = os.path.join('sb_mnist2', split, 'coarse', str(coarse_digit), img_name)
        img.save(coarse_path)

        # 保存到 fine 目录
        fine_path = os.path.join('sb_mnist2', split, 'fine', str(fine_digit), img_name)
        img.save(fine_path)

# 生成训练和测试图像
generate_images(120000, 'train')
generate_images(20000, 'test')
