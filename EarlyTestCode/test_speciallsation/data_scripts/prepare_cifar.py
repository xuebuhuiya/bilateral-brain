import os
import cv2
import numpy as np
import os.path as osp
from tqdm import tqdm

''' 
This script reads from the cifar pickle 
and writes the images into appropriately named folders
for coarse and fine labels for both the train and test splits

Note that the same image is written to two locations, one for each label type 
because pytorch gets the name of the label from the folder
'''

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
      dict = pickle.load(fo, encoding='bytes')
    return dict

# 该函数用于从pickle文件中读取数据，并将其反序列化为Python字典
# pickle是Python中的一个标准库，用于序列化和反序列化Python对象。
# 序列化是指将对象转换为字节流以便存储或传输，
# 反序列化是指将字节流转换回对象。
# 在处理数据时，尤其是像CIFAR-100这样的大规模数据集，使用pickle可以方便地将数据存储为文件，并在需要时快速加载


# set paths here
# ------------------------------------------------------
metadata_path = 'dataset/cifar-100-python/meta' # source cifar 'meta' file元数据文件路径
raw_data_path = 'dataset/cifar-100-python/' # source cifar files here (must have a trailing slash)CIFAR-100数据集的原始文件路径
target_dir = 'datasets_test/CIFAR100'    # the output images go here输出图像保存的目标目录
# ------------------------------------------------------

metadata = unpickle(metadata_path)
superclass_dict = dict(list(enumerate(metadata[b'coarse_label_names'])))
# 读取元数据文件，并将粗标签名称转换为字典格式，其中键是索引，值是粗标签名称

# file paths 设置训练数据和测试数据的文件路径。
data_train_path = raw_data_path + 'train'
data_test_path = raw_data_path + 'test'

# read dictionary 读取训练和测试数据字典
data_train_dict = unpickle(data_train_path)
data_test_dict = unpickle(data_test_path)

# prepare first level of the loop
base_dir_dic = {'train': osp.join(target_dir, 'train'), 'test': osp.join(target_dir, 'test')}
filenames_dic = {'train': data_train_dict[b'filenames'], 'test': data_test_dict[b'filenames']}
data_dic = {'train': data_train_dict[b'data'], 'test': data_test_dict[b'data']}

# base_dir_dic：训练和测试数据的基础目录。
# filenames_dic：训练和测试数据的文件名。
# data_dic：训练和测试数据的图像数据。

# prepare the second level of the loop
coarse_label_train = np.array(data_train_dict[b'coarse_labels'])
fine_label_train = np.array(data_train_dict[b'fine_labels'])
coarse_label_test = np.array(data_test_dict[b'coarse_labels'])
fine_label_test = np.array(data_test_dict[b'fine_labels'])

coarse_label_dic = {'train': coarse_label_train, 'test': coarse_label_test}
fine_label_dic = {'train': fine_label_train, 'test': fine_label_test}
# 将训练和测试数据的粗标签和细标签转换为NumPy数组，并分别存储在字典中。


# for: train, test
for split in ['train', 'test']:
  os.makedirs(base_dir_dic[split], exist_ok=True)

  label_dic_arr = [coarse_label_dic, fine_label_dic]
  label_type_arr = ['coarse', 'fine']
  
  print('----> writing images for {}'.format(split))
  for i in tqdm(range(data_dic[split].shape[0])):
    
    r = data_dic[split][i][:1024].reshape(32, 32)
    g = data_dic[split][i][1024:2048].reshape(32, 32)
    b = data_dic[split][i][2048:].reshape(32, 32)
    img = np.stack([b, g, r]).transpose((1, 2, 0))
    # 对于每张图像，将图像数据从一维数组转换为32x32的三通道图像（RGB）。

    # for: coarse, fine
    for label_dic, label_type in zip(label_dic_arr, label_type_arr):
      out_dir = osp.join(base_dir_dic[split], label_type)
      if not osp.exists(osp.join(out_dir, str(label_dic[split][i]))):
          os.makedirs(osp.join(out_dir, str(label_dic[split][i])))

      cv2.imwrite(osp.join(out_dir, 
          str(label_dic[split][i]), str(filenames_dic[split][i], 'utf-8')), img)
