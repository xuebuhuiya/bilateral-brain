import os
import pickle

# 假设的 coarse 和 fine 类别名称
coarse_label_names = [f'coarse_class_{i}' for i in range(10)]  # coarse 类别名称
fine_label_names = [f'fine_class_{i}' for i in range(10)]      # fine 类别名称

def create_meta_file(output_file):
    meta_data = {
        b'fine_label_names': fine_label_names,   # 细分类名称
        b'coarse_label_names': coarse_label_names  # 粗分类名称
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(meta_data, f)
    print(f"Meta file saved to {output_file}")

def create_data_file(root_dir, output_file):
    data_dict = {
        b'filenames': [],
        b'coarse_labels': [],
        b'data': [],  # 添加 b'data' 键，值为空列表
        # b'fine_labels': [],  # 如果需要 fine_labels，可以取消注释
    }

    root_dir = os.path.abspath(root_dir)

    # 遍历图片目录，生成文件名和 coarse 标签的映射
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith('.png') or filename.endswith('.jpg'):
                # 获取文件的相对路径（相对于 root_dir）
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, root_dir)
                # 使用统一的路径分隔符
                rel_path = rel_path.replace('\\', '/')
                
                # 提取 coarse 标签
                parts = filename.split('_')
                coarse_label = int(parts[0])  # 文件名格式：coarse_fine_id.png

                # 将数据添加到字典
                data_dict[b'filenames'].append(filename.encode('utf-8'))  # 转换为字节字符串
                data_dict[b'coarse_labels'].append(coarse_label)
                data_dict[b'data'].append([])  # 添加占位数据，保持长度一致
                # 如果需要 fine_labels，可以添加
                # fine_label = int(parts[1])
                # data_dict[b'fine_labels'].append(fine_label)
                
    # 将字典保存到 pickle 文件
    with open(output_file, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Data file saved to {output_file}")


# 生成 meta 文件
create_meta_file('sb_mnist41/meta')

# 生成 train 和 test 数据文件
create_data_file('sb_mnist4/train/fine', 'sb_mnist41/train')  # 注意：root_dir 指向 'train/fine'
create_data_file('sb_mnist4/test/fine', 'sb_mnist41/test')    # 同样，指向 'test/fine'
