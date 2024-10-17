import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
import torchvision.models as models
from models.resnet import resnet9
from models.macro import BilateralNet
def register_all_hooks(model):
    features = {}

    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    # 遍历所有子模块并注册hook
    for name, module in model.named_modules():
        # 过滤掉整个模型本身，只对子模块注册hook
        if name != "":
            module.register_forward_hook(get_features(name))

    return features


# 创建模型实例并注册hook
farch = 'resnet9'
fine_k = ''
fine_per_k = ''
main_layer = 0
# model = models.vgg11(weights=None)
# model = globals()[farch](Namespace(**{"k": fine_k, "k_percent": fine_per_k,}))
# features = register_all_hooks(model)
# for name, module in model.named_modules():
#     if main_layer:
#         if '.' not in name:
#             print(name)
#     else:
#         print(name)

model = BilateralNet(mode_out='pred',farch='resnet9', carch='resnet9', fine_k=10, fine_per_k=0.5, coarse_k=5, coarse_per_k=0.5)
# 现在`features`字典将在模型前向传播时填充所有层的输出
print(model)

# import tensorflow as tf
# print(tf.__version__)