
# import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from argparse import Namespace
from models.resnet import resnet9
from models.vgg import vgg11
from models.sparse_resnet import sparse_resnet9
from utils import setup_logger

# from get_feature import register_all_hooks

logger = setup_logger(__name__)

def check_list_mode_head():
    return ['fine', 'coarse', 'both']

def check_list_mode_out():
    return ['pred', 'features']

def check_modes(mode_heads, mode_out):
    if mode_heads and mode_heads not in check_list_mode_head():
        raise ValueError('Mode_head is invalid')
    if mode_out and mode_out not in check_list_mode_out():
        raise ValueError('Mode_out is invalid')

def add_head(num_features, num_classes, dropout):

    logger.debug(f"------- Add head with num_features: {num_features}, num_classes: {num_classes}, dropout: {dropout}")

    mod = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, num_classes))
    return mod

class BilateralNet(nn.Module):
    """
    A Bilateral network with two hemispheres and two heads
    The hemispheres types are configurable.
    The heads are for fine and coarse labels respectively.
    """

    def __init__(self,
                 mode_out: str,
                 farch: str, carch: str,
                 cmodel_path = None, fmodel_path = None,
                 cfreeze_params: bool = True, ffreeze_params: bool = True,
                 fine_k: float = None, fine_per_k: float = None,
                 coarse_k: float = None, coarse_per_k: float = None,
                 dropout: float = 0.0,
                 cc_model: str = None,  # 新增字段
                 learning_type: str = None,  # 新增字段
                 connection_type: str = None,  # 新增字段
                 concat_method: str = None):
        """ Initialize BilateralNet

        Args:
            carch (str): architecture for coarse labels
            farch (str): architecture for fine labels
            mode_out (str): where to get the outputs from
                                both = classifications from both heads
                                feature = features from both hemispheres (input to heads)
        """
        super(BilateralNet, self).__init__()
        
        logger.debug(f"------- Initialize BilaterallNet with farch: {farch}, carch: {carch}, mode_out: {mode_out}, dropout: {dropout}")        
        check_modes(None, mode_out)

        self.mode_out = mode_out
        
        # create the hemispheres
        self.fine_hemi = globals()[farch](Namespace(**{"k": fine_k, "k_percent": fine_per_k,}))
        self.coarse_hemi = globals()[carch](Namespace(**{"k": coarse_k, "k_percent": coarse_per_k,}))

        # self.coarse_hemi = globals()[carch](Namespace(**{"k": coarse_k, "k_percent": coarse_per_k,}))    
        
        self.cc_model = cc_model
        # if cc_model != "none":
        #     self.learning_type = learning_type
        #     self.connection_type = connection_type
        #     self.concat_method = concat_method
        self.learning_type = learning_type
        self.connection_type = connection_type
        self.concat_method = concat_method
        # load the saved trained parameters, and freeze from further training
        if fmodel_path is not None and fmodel_path != '':
            logger.debug("------- Load fine hemisphere")
            load_hemi_model(self.fine_hemi, fmodel_path)
        if cmodel_path is not None and cmodel_path != '':
            logger.debug("------- Load coarse hemisphere")
            load_hemi_model(self.coarse_hemi, cmodel_path)
        
        if ffreeze_params:
            freeze_params(self.fine_hemi)
            logger.debug("      ---> freeze fine")
        if cfreeze_params:
            freeze_params(self.coarse_hemi)
            logger.debug("     ---> freeze coarse")


        # # Freeze parameters, except specific layers
        # if ffreeze_params:
        #     freeze_params2(self.fine_hemi, unfreeze_layers=['conv3', 'res2'])
        #     logger.debug("      ---> freeze fine without conv3 and res2 layers")
        # if cfreeze_params:
        #     freeze_params2(self.coarse_hemi, unfreeze_layers=['conv3', 'res2'])
        #     logger.debug("     ---> freeze coarse without conv3 and res2 layers") 

        # if ffreeze_params:
        #     freeze_params2(self.fine_hemi, unfreeze_layers=['conv3'])
        #     logger.debug("      ---> freeze fine without conv3 layer")
        # if cfreeze_params:
        #     freeze_params2(self.coarse_hemi, unfreeze_layers=['conv3'])
        #     logger.debug("     ---> freeze coarse without conv3 layer")

        # add heads
        num_features = self.fine_hemi.num_features + self.coarse_hemi.num_features
        logger.debug(f"-- num_features: {num_features}")
        # self.fine_head = add_head(num_features, 100, dropout)
        # self.coarse_head = add_head(num_features, 20, dropout)
        self.fine_head = add_head(num_features, 10, dropout)
        self.coarse_head = add_head(num_features, 10, dropout)

        # if self.connection_type == "residual_block" and self.cc_model == "nolearning" :
        if self.connection_type == "residual_block"  :
            logger.debug("     ---> combining model with residual block")
            if self.concat_method == "cat" :
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.adjust1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1).to(self.device)
                self.adjust2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1).to(self.device)
            elif self.learning_type =="attention":
                self.attention_fine_to_coarse = AttentionWeightLayer(channel_size=128)
                self.attention_coarse_to_fine = AttentionWeightLayer(channel_size=128)

                self.attention2_fine_to_coarse = AttentionWeightLayer(channel_size=512)
                self.attention2_coarse_to_fine = AttentionWeightLayer(channel_size=512)
            elif self.learning_type =="gate_control":
                self.gated_unit1 = GatedIntercommunicationUnit(channels=128)
                self.gated_unit2 = GatedIntercommunicationUnit(channels=512)
            # elif self.learning_type =="transformer":
            #     # 对应 res1 的 Transformer 模块 (拼接后的嵌入维度为 256)
            #     self.transformer_res1 = FeatureTransformer(256, 8, 6, 4, 0.1, 500)

            #     # 对应 res2 的 Transformer 模块 (拼接后的嵌入维度为 1024)
            #     self.transformer_res2 = FeatureTransformer(1024, 8, 6, 4, 0.1, 500)

        # Register hooks for 'res1' and 'res2' layers
            self.fine_features = {}
            self.coarse_features = {}

            self.fine_hemi.res1.register_forward_hook(self.save_feature_hook(self.fine_features,'fine_res1'))
            self.coarse_hemi.res1.register_forward_hook(self.save_feature_hook(self.coarse_features,'coarse_res1'))
            self.fine_hemi.res2[0].relu.register_forward_hook(self.save_feature_hook(self.fine_features,'fine_res2_relu'))
            self.coarse_hemi.res2[0].relu.register_forward_hook(self.save_feature_hook(self.coarse_features,'coarse_res2_relu'))       
        # elif self.connection_type == "convolutional_block" and self.cc_model == "nolearning":
        elif self.connection_type == "convolutional_block" :    
            logger.debug("     ---> combining model with convolutional block")
            if self.concat_method == "cat":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # 根据conv2和conv3的输出通道数进行调整
                self.adjust1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1).to(self.device)  # 输入通道数为256
                self.adjust2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1).to(self.device)  # 输入通道数为512
            elif self.learning_type =="attention":
                self.attention_fine_to_coarse = AttentionWeightLayer(channel_size=128)
                self.attention_coarse_to_fine = AttentionWeightLayer(channel_size=128)

                self.attention2_fine_to_coarse = AttentionWeightLayer(channel_size=256)
                self.attention2_coarse_to_fine = AttentionWeightLayer(channel_size=256)
            elif self.learning_type =="gate_control":
                self.gated_unit1 = GatedIntercommunicationUnit2(channels=128)
                self.gated_unit2 = GatedIntercommunicationUnit2(channels=256)
            # 注册钩子函数以捕获conv2和conv3的特征
            self.fine_features = {}
            self.coarse_features = {}

            # 对于conv2, 注册钩子到ReLU后，因为ReLU是conv2序列的最后操作
            self.fine_hemi.conv2[2].register_forward_hook(self.save_feature_hook(self.fine_features,'fine_conv2'))
            self.coarse_hemi.conv2[2].register_forward_hook(self.save_feature_hook(self.coarse_features,'coarse_conv2'))

            # 对于conv3, 注册钩子到ReLU后，同样地
            self.fine_hemi.conv3[2].register_forward_hook(self.save_feature_hook(self.fine_features,'fine_conv3'))
            self.coarse_hemi.conv3[2].register_forward_hook(self.save_feature_hook(self.coarse_features,'coarse_conv3'))

    def forward(self, x):
        """主 forward 方法，根据 mode 动态调用具体的 forward 方法"""
        if self.cc_model == "nolearning":
            if self.connection_type == "residual_block":
                if self.concat_method == "cat":
                    return self.forward_residual_cat(x)
                elif self.concat_method == "add":
                    return self.forward_residual_add(x)
                elif self.concat_method == "halfdata":
                    return self.forward_residual_halfdata(x)
                elif self.concat_method == "ave_pooling":
                    return self.forward_residual_ave_pooling(x)
                else:
                    raise ValueError("Unsupported concat method for residual_block")
            elif self.connection_type == "convolutional_block":
                pass
                # logger.debug("     ---> combining model with convolutional block")
                if self.concat_method == "cat":
                    return self.forward_convolutional_cat(x)
                elif self.concat_method == "add":
                    return self.forward_convolutional_add(x)
                elif self.concat_method == "halfdata":
                    return self.forward_convolutional_halfdata(x)
                elif self.concat_method == "ave_pooling":
                    return self.forward_convolutional_ave_pooling(x)
                else:
                    raise ValueError("Unsupported concat method for convolutional_block")
        elif self.cc_model == "learning":
            if self.connection_type == "residual_block":
                if self.learning_type == "attention":
                    return self.forward_residual_attention(x)
                    
                elif self.learning_type == "gate_control":
                    return self.forward_residual_gate_control(x)
                    
                elif self.learning_type == "transformer":
                    return self.forward_residual_transformer(x)
                    
                else:
                    raise ValueError("Unsupported learning type for residual_block")
            elif self.connection_type == "convolutional_block":
                if self.learning_type == "attention":
                    return self.forward_convolutional_attention(x)
                    
                elif self.learning_type == "gate_control":
                    return self.forward_convolutional_gate_control(x)
                    
                elif self.learning_type == "transformer":
                    # return self.forward_convolutional_transformer(x)
                    pass
                else:
                    raise ValueError("Unsupported learning type for convolutional_block")
        else:
            # logger.debug("     ---> default way to combin model")
            return self.forward_default(x)
    
    
    def forward_default(self, x):
        fembed = self.fine_hemi(x)
        cembed = self.coarse_hemi(x)
        embed = torch.cat([fembed, cembed], axis=1)
        if self.mode_out == 'feature':
            return embed
        elif self.mode_out == 'pred':
            f_out = self.fine_head(embed)
            c_out = self.coarse_head(embed)
            return f_out, c_out

    # def forward(self, x):
    # #It should not work.
    #     fembed = self.fine_hemi(x)
    #     cembed = self.coarse_hemi(x)

    #     # Cross-layer connections and 1x1 convolutions to adjust feature dimensions
    #     if hasattr(self, 'fine_res1') and hasattr(self, 'coarse_res1'):
    #         fine_res1 = self.fine_res1
    #         # print("fine_res1 shape:", fine_res1.shape)
    #         coarse_res1 = self.coarse_res1
    #         # print("coarse_res1 shape:", coarse_res1.shape)
    #         cross1 = torch.cat([fine_res1, coarse_res1], dim=1)
    #         cross1 = self.adjust1(cross1)  # Adjusted feature map
    #         # print("cross1 shape:", cross1.shape)

    #         # cross1 = feature_combine(fine_res1, coarse_res1, mode='cat', adjust_layer=self.adjust1)

    #     if hasattr(self, 'fine_res2_relu') and hasattr(self, 'coarse_res2_relu'):
    #         fine_res2_relu = self.fine_res2_relu
    #         # print("fine_res2_relu shape:", fine_res2_relu.shape)
    #         coarse_res2_relu = self.coarse_res2_relu
    #         # print("coarse_res2_relu shape:", coarse_res2_relu.shape)
    #         cross2 = torch.cat([fine_res2_relu, coarse_res2_relu], dim=1)
    #         # print("cross2 shape before adjust2:", cross2.shape)
    #         cross2 = self.adjust2(cross2)  # Adjusted feature map
    #         # print("cross2 shape:", cross2.shape)

    #     embed = torch.cat([fembed, cembed], dim=1)
        
    #     if self.mode_out == 'feature':
    #         return embed  # or return modified embed with cross1, cross2
    #     elif self.mode_out == 'pred':
    #         f_out = self.fine_head(embed)
    #         c_out = self.coarse_head(embed)
    #         return f_out, c_out


    def forward_residual_cat(self, x):
        # 通过fine_hemi和coarse_hemi获取初始嵌入特征
        fembed = self.fine_hemi.conv1(x)
        cembed = self.coarse_hemi.conv1(x)
        
        fembed = self.fine_hemi.conv2(fembed)
        cembed = self.coarse_hemi.conv2(cembed)
        
        # 在res1层后进行特征拼接和调整
        fembed = self.fine_hemi.res1(fembed)
        cembed = self.coarse_hemi.res1(cembed)
        # if hasattr(self, 'fine_res1') and hasattr(self, 'coarse_res1'):
        #     # print("Using intermediate features from res1")
        #     fine_res1 = self.fine_res1
        #     coarse_res1 = self.coarse_res1
        #     cross1 = torch.cat([fine_res1, coarse_res1], dim=1)
        #     cross1 = self.adjust1(cross1)

        #     # cross1 = self.pool_and_combine(fine_res1, coarse_res1)

        #     fembed = cross1
        #     cembed = cross1
        fine_res1 = self.fine_features['fine_res1']
        coarse_res1 = self.coarse_features['coarse_res1']
        cross1 = torch.cat([fine_res1, coarse_res1], dim=1)
        cross1 = self.adjust1(cross1)

        # 更新特征
        fembed = cross1
        cembed = cross1.clone()

        # 在conv3层进行处理
        fembed = self.fine_hemi.conv3(fembed)
        cembed = self.coarse_hemi.conv3(cembed)

        # 在conv4层进行处理
        fembed = self.fine_hemi.conv4(fembed)
        cembed = self.coarse_hemi.conv4(cembed)
        
        # 在res2层后进行特征拼接和调整
        fembed = self.fine_hemi.res2[0](fembed)
        cembed = self.coarse_hemi.res2[0](cembed)

        # if hasattr(self, 'fine_res2_relu') and hasattr(self, 'coarse_res2_relu'):
        #     # print("Using intermediate features from res2")
        #     fine_res2_relu = self.fine_res2_relu
        #     coarse_res2_relu = self.coarse_res2_relu
        #     cross2 = torch.cat([fine_res2_relu, coarse_res2_relu], dim=1)
        #     cross2 = self.adjust2(cross2)

        #     # cross2 = self.pool_and_combine(fine_res2_relu, coarse_res2_relu)

        #     fembed = cross2
        #     cembed = cross2
        # 使用从res2_relu层捕获的特征进行拼接和调整
        fine_res2_relu = self.fine_features['fine_res2_relu']
        coarse_res2_relu = self.coarse_features['coarse_res2_relu']
        cross2 = torch.cat([fine_res2_relu, coarse_res2_relu], dim=1)
        cross2 = self.adjust2(cross2)

        # 更新特征
        fembed = cross2
        cembed = cross2.clone()

        # 处理res2剩余的层
        fembed = self.fine_hemi.res2[1](fembed)
        cembed = self.coarse_hemi.res2[1](cembed)
        fembed = self.fine_hemi.res2[2](fembed)
        cembed = self.coarse_hemi.res2[2](cembed)
        
        # # 在res2层进行处理，不再进行特征拼接
        # fembed = self.fine_hemi.res2(fembed)
        # cembed = self.coarse_hemi.res2(cembed)

        # 拼接最终的嵌入特征，保持空间结构
        embed = torch.cat([fembed, cembed], dim=1)
        
        if self.mode_out == 'feature':
            return embed
        elif self.mode_out == 'pred':
            f_out = self.fine_head(embed)
            c_out = self.coarse_head(embed)
            return f_out, c_out
             
    def forward_residual_add(self, x):
    # 通过fine_hemi和coarse_hemi获取初始嵌入特征
        fembed = self.fine_hemi.conv1(x)
        cembed = self.coarse_hemi.conv1(x)
        
        fembed = self.fine_hemi.conv2(fembed)
        cembed = self.coarse_hemi.conv2(cembed)
        
        # 在res1层后进行特征加权和
        fembed = self.fine_hemi.res1(fembed)
        cembed = self.coarse_hemi.res1(cembed)
        # if hasattr(self, 'fine_res1') and hasattr(self, 'coarse_res1'):
        #     fine_res1 = self.fine_res1
        #     coarse_res1 = self.coarse_res1
        #     cross1 = 0.5 * fine_res1 + 0.5 * coarse_res1  # 初始权重设置为0.5

        #     fembed = cross1
        #     cembed = cross1
        fine_res1 = self.fine_features['fine_res1']
        coarse_res1 = self.coarse_features['coarse_res1']
        cross1 = 0.5 * fine_res1 + 0.5 * coarse_res1
        fembed = cross1
        cembed = cross1.clone()

        # 在conv3层进行处理
        fembed = self.fine_hemi.conv3(fembed)
        cembed = self.coarse_hemi.conv3(cembed)

        # 在conv4层进行处理
        fembed = self.fine_hemi.conv4(fembed)
        cembed = self.coarse_hemi.conv4(cembed)
        
        # 在res2层后进行特征加权和
        fembed = self.fine_hemi.res2[0](fembed)
        cembed = self.coarse_hemi.res2[0](cembed)
        # if hasattr(self, 'fine_res2_relu') and hasattr(self, 'coarse_res2_relu'):
        #     fine_res2_relu = self.fine_res2_relu
        #     coarse_res2_relu = self.coarse_res2_relu
        #     cross2 = 0.5 * fine_res2_relu + 0.5 * coarse_res2_relu  # 初始权重设置为0.5

        #     fembed = cross2
        #     cembed = cross2
        fine_res2_relu = self.fine_features['fine_res2_relu']
        coarse_res2_relu = self.coarse_features['coarse_res2_relu']
        cross2 = 0.5 * fine_res2_relu + 0.5 * coarse_res2_relu

        # 更新特征
        fembed = cross2
        cembed = cross2.clone()

        # 处理res2剩余的层
        fembed = self.fine_hemi.res2[1](fembed)
        cembed = self.coarse_hemi.res2[1](cembed)
        fembed = self.fine_hemi.res2[2](fembed)
        cembed = self.coarse_hemi.res2[2](cembed)
        
        # 拼接最终的嵌入特征，保持空间结构
        embed = torch.cat([fembed, cembed], dim=1)
        
        if self.mode_out == 'feature':
            return embed
        elif self.mode_out == 'pred':
            f_out = self.fine_head(embed)
            c_out = self.coarse_head(embed)
            return f_out, c_out

    def forward_residual_halfdata(self, x):
        # 初始特征提取
        fembed = self.fine_hemi.conv1(x)
        cembed = self.coarse_hemi.conv1(x)
        
        fembed = self.fine_hemi.conv2(fembed)
        cembed = self.coarse_hemi.conv2(cembed)
        
        # 在res1层后进行特征交叉更新
        fembed = self.fine_hemi.res1(fembed)
        cembed = self.coarse_hemi.res1(cembed)

        # 使用捕获的特征进行交叉更新
        # if hasattr(self, 'fine_res1') and hasattr(self, 'coarse_res1'):
        #     # 交叉更新，确保只使用50%的数据量
        #     size = self.fine_res1.shape[1] // 2  # 获取通道数的一半
        #     # cross_fine = torch.cat((self.fine_res1[:, :size], self.coarse_res1[:, :size]), dim=1)
        #     # cross_coarse = torch.cat((self.coarse_res1[:, :size], self.fine_res1[:, :size]), dim=1)
        #     cross_fine = 0.5 * self.fine_res1[:, :size] + 0.5 * self.coarse_res1[:, :size]
        #     cross_coarse = 0.5 * self.coarse_res1[:, :size] + 0.5 * self.fine_res1[:, :size]
            
        #     fembed[:, :size] = cross_fine
        #     cembed[:, :size] = cross_coarse

        fine_res1 = self.fine_features['fine_res1']
        coarse_res1 = self.coarse_features['coarse_res1']
        size = fine_res1.shape[1] // 2
        cross_fine = 0.5 * fine_res1[:, :size] + 0.5 * coarse_res1[:, :size]
        cross_coarse = 0.5 * coarse_res1[:, :size] + 0.5 * fine_res1[:, :size]
            
        fembed[:, :size] = cross_fine
        cembed[:, :size] = cross_coarse
        # 继续通过网络的其它层
        fembed = self.fine_hemi.conv3(fembed)
        cembed = self.coarse_hemi.conv3(cembed)

        fembed = self.fine_hemi.conv4(fembed)
        cembed = self.coarse_hemi.conv4(cembed)
        
        # 在res2层后再次进行特征交叉更新
        fembed = self.fine_hemi.res2[0](fembed)
        cembed = self.coarse_hemi.res2[0](cembed)

        # if hasattr(self, 'fine_res2_relu') and hasattr(self, 'coarse_res2_relu'):
        #     size = self.fine_res2_relu.shape[1] // 2
        #     # cross_fine = torch.cat((self.fine_res2_relu[:, :size], self.coarse_res2_relu[:, :size]), dim=1)
        #     # cross_coarse = torch.cat((self.coarse_res2_relu[:, :size], self.fine_res2_relu[:, :size]), dim=1)
        #     cross_fine = 0.5 * self.fine_res2_relu[:, :size] + 0.5 * self.coarse_res2_relu[:, :size]
        #     cross_coarse = 0.5 * self.coarse_res2_relu[:, :size] + 0.5 * self.fine_res2_relu[:, :size]
            
        #     fembed[:, :size] = cross_fine
        #     cembed[:, :size] = cross_coarse

        fine_res2_relu = self.fine_features['fine_res2_relu']
        coarse_res2_relu = self.coarse_features['coarse_res2_relu']
        size = fine_res2_relu.shape[1] // 2
        cross_fine = 0.5 * fine_res2_relu[:, :size] + 0.5 * coarse_res2_relu[:, :size]
        cross_coarse = 0.5 * coarse_res2_relu[:, :size] + 0.5 * fine_res2_relu[:, :size]
            
        fembed[:, :size] = cross_fine
        cembed[:, :size] = cross_coarse

        # 处理res2剩余的层
        fembed = self.fine_hemi.res2[1](fembed)
        cembed = self.coarse_hemi.res2[1](cembed)
        fembed = self.fine_hemi.res2[2](fembed)
        cembed = self.coarse_hemi.res2[2](cembed)   

        # 最后的层处理和输出
        embed = torch.cat([fembed, cembed], dim=1)
        
        if self.mode_out == 'feature':
            return embed
        elif self.mode_out == 'pred':
            f_out = self.fine_head(embed)
            c_out = self.coarse_head(embed)
            return f_out, c_out

    def forward_residual_ave_pooling(self, x):
        # 初始特征提取
        fembed = self.fine_hemi.conv1(x)
        cembed = self.coarse_hemi.conv1(x)
        
        fembed = self.fine_hemi.conv2(fembed)
        cembed = self.coarse_hemi.conv2(cembed)
        
        # 在res1层后进行特征交叉更新
        fembed = self.fine_hemi.res1(fembed)
        cembed = self.coarse_hemi.res1(cembed)

        # # 使用池化和组合进行特征融合
        # if hasattr(self, 'fine_res1') and hasattr(self, 'coarse_res1'):
        #     fembed = self.pool_and_combine(self.fine_res1, self.coarse_res1)
        #     cembed = fembed  # 使两边的特征相同

        fine_res1 = self.fine_features['fine_res1']
        coarse_res1 = self.coarse_features['coarse_res1']
        cross1 = self.pool_and_combine(fine_res1, coarse_res1)
        fembed = cross1
        cembed = cross1.clone()

        # 继续通过网络的其它层
        fembed = self.fine_hemi.conv3(fembed)
        cembed = self.coarse_hemi.conv3(cembed)

        fembed = self.fine_hemi.conv4(fembed)
        cembed = self.coarse_hemi.conv4(cembed)
        
        # 在res2层后再次进行特征交叉更新
        fembed = self.fine_hemi.res2[0](fembed)
        cembed = self.coarse_hemi.res2[0](cembed)

        if hasattr(self, 'fine_res2_relu') and hasattr(self, 'coarse_res2_relu'):
            fembed = self.pool_and_combine(self.fine_res2_relu, self.coarse_res2_relu)
            cembed = fembed  # 使两边的特征相同

        fine_res2_relu = self.fine_features['fine_res2_relu']
        coarse_res2_relu = self.coarse_features['coarse_res2_relu']
        cross2 = self.pool_and_combine(fine_res2_relu, coarse_res2_relu)
        # 更新特征
        fembed = cross2
        cembed = cross2.clone()

        # 处理res2剩余的层
        fembed = self.fine_hemi.res2[1](fembed)
        cembed = self.coarse_hemi.res2[1](cembed)
        fembed = self.fine_hemi.res2[2](fembed)
        cembed = self.coarse_hemi.res2[2](cembed)   

        # 最后的层处理和输出
        embed = torch.cat([fembed, cembed], dim=1)
        
        if self.mode_out == 'feature':
            return embed
        elif self.mode_out == 'pred':
            f_out = self.fine_head(embed)
            c_out = self.coarse_head(embed)
            return f_out, c_out

    def forward_convolutional_cat(self, x):
        # 通过fine_hemi和coarse_hemi获取初始嵌入特征
        fembed = self.fine_hemi.conv1(x)
        cembed = self.coarse_hemi.conv1(x)
        
        # 在conv2层后进行特征拼接和调整
        fembed = self.fine_hemi.conv2(fembed)
        cembed = self.coarse_hemi.conv2(cembed)
        # if hasattr(self, 'fine_conv2') and hasattr(self, 'coarse_conv2'):
        #     fine_conv2 = self.fine_conv2
        #     coarse_conv2 = self.coarse_conv2
        #     cross1 = torch.cat([fine_conv2, coarse_conv2], dim=1)
        #     cross1 = self.adjust1(cross1)
            
            
        cross1 = torch.cat([self.fine_features['fine_conv2'], self.coarse_features['coarse_conv2']], dim=1)
        cross1 = self.adjust1(cross1)

        fembed = cross1
        cembed = cross1
        fembed = self.fine_hemi.conv2[3](fembed)
        cembed = self.coarse_hemi.conv2[3](cembed)
        # 处理res1层
        fembed = self.fine_hemi.res1(fembed)
        cembed = self.coarse_hemi.res1(cembed)
        
        # 在conv3层后进行特征拼接和调整
        fembed = self.fine_hemi.conv3(fembed)
        cembed = self.coarse_hemi.conv3(cembed)
        # if hasattr(self, 'fine_conv3') and hasattr(self, 'coarse_conv3'):
        #     fine_conv3 = self.fine_conv3
        #     coarse_conv3 = self.coarse_conv3
        #     cross2 = torch.cat([fine_conv3, coarse_conv3], dim=1)
        #     cross2 = self.adjust2(cross2)
        cross2 = torch.cat([self.fine_features['fine_conv3'], self.coarse_features['coarse_conv3']], dim=1)
        cross2 = self.adjust2(cross2)   
        fembed = cross2
        cembed = cross2
        fembed = self.fine_hemi.conv3[3](fembed)
        cembed = self.coarse_hemi.conv3[3](cembed)
        # 在conv4层进行处理
        fembed = self.fine_hemi.conv4(fembed)
        cembed = self.coarse_hemi.conv4(cembed)
        # 处理res2层
        fembed = self.fine_hemi.res2(fembed)
        cembed = self.coarse_hemi.res2(cembed)
           
        # 拼接最终的嵌入特征，保持空间结构
        embed = torch.cat([fembed, cembed], dim=1)
        
        if self.mode_out == 'feature':
            return embed
        elif self.mode_out == 'pred':
            f_out = self.fine_head(embed)
            c_out = self.coarse_head(embed)
            return f_out, c_out

    def forward_convolutional_add(self, x):
        # 通过fine_hemi和coarse_hemi获取初始嵌入特征
        fembed = self.fine_hemi.conv1(x)
        cembed = self.coarse_hemi.conv1(x)
        
        # 在conv2层后进行特征加权和
        fembed = self.fine_hemi.conv2(fembed)
        cembed = self.coarse_hemi.conv2(cembed)
        fine_conv2 = self.fine_features['fine_conv2']
        coarse_conv2 = self.coarse_features['coarse_conv2']
        cross1 = 0.5 * fine_conv2 + 0.5 * coarse_conv2  # 初始权重设置为0.5
        fembed = cross1
        cembed = cross1
        fembed = self.fine_hemi.conv2[3](fembed)
        cembed = self.coarse_hemi.conv2[3](cembed)

        # 处理res1层
        fembed = self.fine_hemi.res1(fembed)
        cembed = self.coarse_hemi.res1(cembed)
        
        # 在conv3层后进行特征加权和
        fembed = self.fine_hemi.conv3(fembed)
        cembed = self.coarse_hemi.conv3(cembed)
        fine_conv3 = self.fine_features['fine_conv3']
        coarse_conv3 = self.coarse_features['coarse_conv3']
        cross2 = 0.5 * fine_conv3 + 0.5 * coarse_conv3  # 初始权重设置为0.5

        fembed = cross2
        cembed = cross2
        fembed = self.fine_hemi.conv3[3](fembed)
        cembed = self.coarse_hemi.conv3[3](cembed)

        # 在conv4层进行处理
        fembed = self.fine_hemi.conv4(fembed)
        cembed = self.coarse_hemi.conv4(cembed)

        # 处理res2层
        fembed = self.fine_hemi.res2(fembed)
        cembed = self.coarse_hemi.res2(cembed)
        
        # 拼接最终的嵌入特征，保持空间结构
        embed = torch.cat([fembed, cembed], dim=1)
        
        if self.mode_out == 'feature':
            return embed
        elif self.mode_out == 'pred':
            f_out = self.fine_head(embed)
            c_out = self.coarse_head(embed)
            return f_out, c_out

    def forward_convolutional_halfdata(self, x):
        # 初始特征提取
        fembed = self.fine_hemi.conv1(x)
        cembed = self.coarse_hemi.conv1(x)
        
        fembed = self.fine_hemi.conv2(fembed)
        cembed = self.coarse_hemi.conv2(cembed)
        
        # # 在conv2层后进行特征交叉更新
        # if hasattr(self, 'fine_conv2') and hasattr(self, 'coarse_conv2'):
        #     size = self.fine_conv2.shape[1] // 2  # 获取通道数的一半
        #     cross_fine = 0.5 * self.fine_conv2[:, :size] + 0.5 * self.coarse_conv2[:, :size]
        #     cross_coarse = 0.5 * self.coarse_conv2[:, :size] + 0.5 * self.fine_conv2[:, :size]
            
        #     fembed[:, :size] = cross_fine
        #     cembed[:, :size] = cross_coarse
        # print(f"After conv2 - fembed shape: {fembed.shape}, cembed shape: {cembed.shape}")
        fine_conv2 = self.fine_features['fine_conv2']
        coarse_conv2 = self.coarse_features['coarse_conv2']
        size = fine_conv2.shape[1] // 2  # 获取通道数的一半
        cross_fine = 0.5 * fine_conv2[:, :size] + 0.5 * coarse_conv2[:, :size]
        cross_coarse = 0.5 * coarse_conv2[:, :size] + 0.5 * fine_conv2[:, :size]
        cross_fine = torch.nn.functional.interpolate(cross_fine, size=fembed.shape[2:])
        cross_coarse = torch.nn.functional.interpolate(cross_coarse, size=cembed.shape[2:])
        # print(f"Cross feature shapes after conv2 - cross_fine shape: {cross_fine.shape}, cross_coarse shape: {cross_coarse.shape}")
        fembed[:, :size] = cross_fine
        cembed[:, :size] = cross_coarse
        fembed = self.fine_hemi.conv2[3](fembed)
        cembed = self.coarse_hemi.conv2[3](cembed)

        # 处理res1层
        fembed = self.fine_hemi.res1(fembed)
        cembed = self.coarse_hemi.res1(cembed)
        
        # 在conv3层后进行特征交叉更新
        fembed = self.fine_hemi.conv3(fembed)
        cembed = self.coarse_hemi.conv3(cembed)
        
        # if hasattr(self, 'fine_conv3') and hasattr(self, 'coarse_conv3'):
        #     size = self.fine_conv3.shape[1] // 2  # 获取通道数的一半
        #     cross_fine = 0.5 * self.fine_conv3[:, :size] + 0.5 * self.coarse_conv3[:, :size]
        #     cross_coarse = 0.5 * self.coarse_conv3[:, :size] + 0.5 * self.fine_conv3[:, :size]
            
        #     fembed[:, :size] = cross_fine
        #     cembed[:, :size] = cross_coarse
        # print(f"After conv3 - fembed shape: {fembed.shape}, cembed shape: {cembed.shape}")
        fine_conv3 = self.fine_features['fine_conv3']
        coarse_conv3 = self.coarse_features['coarse_conv3']
        size = fine_conv3.shape[1] // 2
        cross_fine = 0.5 * fine_conv3[:, :size] + 0.5 * coarse_conv3[:, :size]
        cross_coarse = 0.5 * coarse_conv3[:, :size] + 0.5 * fine_conv3[:, :size]

        cross_fine = torch.nn.functional.interpolate(cross_fine, size=fembed.shape[2:])
        cross_coarse = torch.nn.functional.interpolate(cross_coarse, size=cembed.shape[2:])
        # print(f"Cross feature shapes after conv3 - cross_fine shape: {cross_fine.shape}, cross_coarse shape: {cross_coarse.shape}")
        fembed[:, :size] = cross_fine
        cembed[:, :size] = cross_coarse
        fembed = self.fine_hemi.conv3[3](fembed)
        cembed = self.coarse_hemi.conv3[3](cembed)
        
        
        # 在conv4层后再次进行特征交叉更新
        fembed = self.fine_hemi.conv4(fembed)
        cembed = self.coarse_hemi.conv4(cembed)

        # 处理res2层
        fembed = self.fine_hemi.res2(fembed)
        cembed = self.coarse_hemi.res2(cembed)

        # 拼接最终的嵌入特征，保持空间结构
        embed = torch.cat([fembed, cembed], dim=1)
        
        if self.mode_out == 'feature':
            return embed
        elif self.mode_out == 'pred':
            f_out = self.fine_head(embed)
            c_out = self.coarse_head(embed)
            return f_out, c_out

    def forward_convolutional_ave_pooling(self, x):
        # 初始特征提取
        fembed = self.fine_hemi.conv1(x)
        cembed = self.coarse_hemi.conv1(x)
        
        fembed = self.fine_hemi.conv2(fembed)
        cembed = self.coarse_hemi.conv2(cembed)
        
        # # 在conv2层后进行特征融合
        # if hasattr(self, 'fine_conv2') and hasattr(self, 'coarse_conv2'):
        #     fembed = self.pool_and_combine(self.fine_conv2, self.coarse_conv2)
        #     cembed = fembed  # 使两边的特征相同
        fine_conv2 = self.fine_features['fine_conv2']
        coarse_conv2 = self.coarse_features['coarse_conv2']
        cross1 = self.pool_and_combine(fine_conv2, coarse_conv2)  
        fembed = cross1
        cembed = cross1
        fembed = self.fine_hemi.conv2[3](fembed)
        cembed = self.coarse_hemi.conv2[3](cembed)

        # 处理res1层
        fembed = self.fine_hemi.res1(fembed)
        cembed = self.coarse_hemi.res1(cembed)
        
        # 在conv3层后再次进行特征融合
        fembed = self.fine_hemi.conv3(fembed)
        cembed = self.coarse_hemi.conv3(cembed)
        
        # if hasattr(self, 'fine_conv3') and hasattr(self, 'coarse_conv3'):
        #     fembed = self.pool_and_combine(self.fine_conv3, self.coarse_conv3)
        #     cembed = fembed  # 使两边的特征相同
        fine_conv3 = self.fine_features['fine_conv3']
        coarse_conv3 = self.coarse_features['coarse_conv3']
        cross2 = self.pool_and_combine(fine_conv3, coarse_conv3)  
        fembed = cross2
        cembed = cross2
        fembed = self.fine_hemi.conv3[3](fembed)
        cembed = self.coarse_hemi.conv3[3](cembed)

        # 在conv4层进行处理
        fembed = self.fine_hemi.conv4(fembed)
        cembed = self.coarse_hemi.conv4(cembed)
        
        # 处理res2层
        fembed = self.fine_hemi.res2(fembed)
        cembed = self.coarse_hemi.res2(cembed)

        # 拼接最终的嵌入特征，保持空间结构
        embed = torch.cat([fembed, cembed], dim=1)
        
        if self.mode_out == 'feature':
            return embed
        elif self.mode_out == 'pred':
            f_out = self.fine_head(embed)
            c_out = self.coarse_head(embed)
            return f_out, c_out

    def forward_residual_attention(self, x):
        # 通过fine_hemi和coarse_hemi获取初始嵌入特征
        fembed = self.fine_hemi.conv1(x)
        cembed = self.coarse_hemi.conv1(x)
        
        fembed = self.fine_hemi.conv2(fembed)
        cembed = self.coarse_hemi.conv2(cembed)
        
        # 在res1层后进行特征加权和
        fembed = self.fine_hemi.res1(fembed)
        cembed = self.coarse_hemi.res1(cembed)

        fine_res1 = self.fine_features['fine_res1']
        coarse_res1 = self.coarse_features['coarse_res1']
        adjusted_fembed_res1 = self.attention_coarse_to_fine(fine_res1, coarse_res1)
        adjusted_cembed_res1 = self.attention_fine_to_coarse(coarse_res1, fine_res1)
        # cross1 = 0.5 * fine_res1 + 0.5 * coarse_res1
        fembed = adjusted_fembed_res1
        cembed = adjusted_cembed_res1

        # 在conv3层进行处理
        fembed = self.fine_hemi.conv3(fembed)
        cembed = self.coarse_hemi.conv3(cembed)

        # 在conv4层进行处理
        fembed = self.fine_hemi.conv4(fembed)
        cembed = self.coarse_hemi.conv4(cembed)
        
        # 在res2层后进行特征加权和
        fembed = self.fine_hemi.res2[0](fembed)
        cembed = self.coarse_hemi.res2[0](cembed)
        
        fine_res2_relu = self.fine_features['fine_res2_relu']
        coarse_res2_relu = self.coarse_features['coarse_res2_relu']
        adjusted_fembed_res2 = self.attention2_coarse_to_fine(fine_res2_relu, coarse_res2_relu)
        adjusted_cembed_res2 = self.attention2_fine_to_coarse(coarse_res2_relu, fine_res2_relu)

        # cross2 = 0.5 * fine_res2_relu + 0.5 * coarse_res2_relu

        # 更新特征
        fembed = adjusted_fembed_res2
        cembed = adjusted_cembed_res2

        # 处理res2剩余的层
        fembed = self.fine_hemi.res2[1](fembed)
        cembed = self.coarse_hemi.res2[1](cembed)
        fembed = self.fine_hemi.res2[2](fembed)
        cembed = self.coarse_hemi.res2[2](cembed)
        
        # 拼接最终的嵌入特征，保持空间结构
        embed = torch.cat([fembed, cembed], dim=1)
        
        if self.mode_out == 'feature':
            return embed
        elif self.mode_out == 'pred':
            f_out = self.fine_head(embed)
            c_out = self.coarse_head(embed)
            return f_out, c_out

    def forward_residual_gate_control(self, x):
        # 通过fine_hemi和coarse_hemi获取初始嵌入特征
        fembed = self.fine_hemi.conv1(x)
        cembed = self.coarse_hemi.conv1(x)
        
        fembed = self.fine_hemi.conv2(fembed)
        cembed = self.coarse_hemi.conv2(cembed)
        
        # 在res1层后进行门控调整
        fembed = self.fine_hemi.res1(fembed)
        cembed = self.coarse_hemi.res1(cembed)

        # 使用门控单元进行特征调整
        fine_res1 = self.fine_features['fine_res1']
        coarse_res1 = self.coarse_features['coarse_res1']
        fembed, cembed = self.gated_unit1(fine_res1, coarse_res1)

        # 在conv3层进行处理
        fembed = self.fine_hemi.conv3(fembed)
        cembed = self.coarse_hemi.conv3(cembed)

        # 在conv4层进行处理
        fembed = self.fine_hemi.conv4(fembed)
        cembed = self.coarse_hemi.conv4(cembed)
        
        # 在res2层后进行门控调整
        fembed = self.fine_hemi.res2[0](fembed)
        cembed = self.coarse_hemi.res2[0](cembed)
        
        # 使用第二个门控单元进行特征调整
        fine_res2_relu = self.fine_features['fine_res2_relu']
        coarse_res2_relu = self.coarse_features['coarse_res2_relu']
        fembed, cembed = self.gated_unit2(fine_res2_relu, coarse_res2_relu)

        # 处理res2剩余的层
        fembed = self.fine_hemi.res2[1](fembed)
        cembed = self.coarse_hemi.res2[1](cembed)
        fembed = self.fine_hemi.res2[2](fembed)
        cembed = self.coarse_hemi.res2[2](cembed)
        
        # 拼接最终的嵌入特征，保持空间结构
        embed = torch.cat([fembed, cembed], dim=1)
        
        if self.mode_out == 'feature':
            return embed
        elif self.mode_out == 'pred':
            f_out = self.fine_head(embed)
            c_out = self.coarse_head(embed)
            return f_out, c_out

    def forward_residual_transformer(self, x):
        # 获取初始嵌入特征
        fembed = self.fine_hemi.conv1(x)
        cembed = self.coarse_hemi.conv1(x)
        
        fembed = self.fine_hemi.conv2(fembed)
        cembed = self.coarse_hemi.conv2(cembed)
        
        # 在res1层后进行特征传递
        fembed = self.fine_hemi.res1(fembed)
        cembed = self.coarse_hemi.res1(cembed)

        fine_res1 = self.fine_features['fine_res1']
        coarse_res1 = self.coarse_features['coarse_res1']

        # 将 fine 和 coarse 特征拼接作为输入
        combined_res1 = torch.cat([fine_res1, coarse_res1], dim=1)
        
        # 使用 Transformer 处理拼接的特征
        combined_res1_transformed = self.transformer_res1(combined_res1, None)
        
        # 分割 Transformer 输出
        fembed_res1, cembed_res1 = torch.chunk(combined_res1_transformed, 2, dim=1)

        # 在conv3层进行处理
        fembed = self.fine_hemi.conv3(fembed_res1)
        cembed = self.coarse_hemi.conv3(cembed_res1)

        # 在conv4层进行处理
        fembed = self.fine_hemi.conv4(fembed)
        cembed = self.coarse_hemi.conv4(cembed)
        
        # 在res2层后进行特征传递
        fembed = self.fine_hemi.res2[0](fembed)
        cembed = self.coarse_hemi.res2[0](cembed)
        
        fine_res2_relu = self.fine_features['fine_res2_relu']
        coarse_res2_relu = self.coarse_features['coarse_res2_relu']

        # 将 fine 和 coarse 特征拼接作为输入
        combined_res2 = torch.cat([fine_res2_relu, coarse_res2_relu], dim=1)
        
        # 使用 Transformer 处理拼接的特征
        combined_res2_transformed = self.transformer_res2(combined_res2, None)
        
        # 分割 Transformer 输出
        fembed_res2, cembed_res2 = torch.chunk(combined_res2_transformed, 2, dim=1)

        # 处理res2剩余的层
        fembed = self.fine_hemi.res2[1](fembed_res2)
        cembed = self.coarse_hemi.res2[1](cembed_res2)
        fembed = self.fine_hemi.res2[2](fembed)
        cembed = self.coarse_hemi.res2[2](cembed)
        
        # 拼接最终的嵌入特征，保持空间结构
        embed = torch.cat([fembed, cembed], dim=1)
        
        if self.mode_out == 'feature':
            return embed
        elif self.mode_out == 'pred':
            f_out = self.fine_head(embed)
            c_out = self.coarse_head(embed)
            return f_out, c_out

    def forward_convolutional_attention(self, x):
        # 通过fine_hemi和coarse_hemi获取初始嵌入特征
        fembed = self.fine_hemi.conv1(x)
        cembed = self.coarse_hemi.conv1(x)
        
        # 在conv2层后进行特征加权和
        fembed = self.fine_hemi.conv2(fembed)
        cembed = self.coarse_hemi.conv2(cembed)
        fine_conv2 = self.fine_features['fine_conv2']
        coarse_conv2 = self.coarse_features['coarse_conv2']
        adjusted_fembed_conv2 = self.attention_coarse_to_fine(fine_conv2, coarse_conv2)
        adjusted_cembed_conv2 = self.attention_fine_to_coarse(coarse_conv2, fine_conv2)
        # cross1 = 0.5 * fine_conv2 + 0.5 * coarse_conv2  # 初始权重设置为0.5
        fembed = adjusted_fembed_conv2 
        cembed = adjusted_cembed_conv2
        fembed = self.fine_hemi.conv2[3](fembed)
        cembed = self.coarse_hemi.conv2[3](cembed)

        # 处理res1层
        fembed = self.fine_hemi.res1(fembed)
        cembed = self.coarse_hemi.res1(cembed)
        
        # 在conv3层后进行特征加权和
        fembed = self.fine_hemi.conv3(fembed)
        cembed = self.coarse_hemi.conv3(cembed)
        fine_conv3 = self.fine_features['fine_conv3']
        coarse_conv3 = self.coarse_features['coarse_conv3']
        # cross2 = 0.5 * fine_conv3 + 0.5 * coarse_conv3  # 初始权重设置为0.5
        adjusted_fembed_conv3 = self.attention2_coarse_to_fine(fine_conv3, coarse_conv3)
        adjusted_cembed_conv3 = self.attention2_fine_to_coarse(coarse_conv3, fine_conv3)
        fembed = adjusted_fembed_conv3
        cembed = adjusted_cembed_conv3
        fembed = self.fine_hemi.conv3[3](fembed)
        cembed = self.coarse_hemi.conv3[3](cembed)

        # 在conv4层进行处理
        fembed = self.fine_hemi.conv4(fembed)
        cembed = self.coarse_hemi.conv4(cembed)

        # 处理res2层
        fembed = self.fine_hemi.res2(fembed)
        cembed = self.coarse_hemi.res2(cembed)
        
        # 拼接最终的嵌入特征，保持空间结构
        embed = torch.cat([fembed, cembed], dim=1)
        
        if self.mode_out == 'feature':
            return embed
        elif self.mode_out == 'pred':
            f_out = self.fine_head(embed)
            c_out = self.coarse_head(embed)
            return f_out, c_out

    def forward_convolutional_gate_control(self, x):
        # 通过fine_hemi和coarse_hemi获取初始嵌入特征
        fembed = self.fine_hemi.conv1(x)
        cembed = self.coarse_hemi.conv1(x)
        
        # 在conv2层后进行特征加权和
        fembed = self.fine_hemi.conv2(fembed)
        cembed = self.coarse_hemi.conv2(cembed)
        fine_conv2 = self.fine_features['fine_conv2']
        coarse_conv2 = self.coarse_features['coarse_conv2']
        fembed, cembed = self.gated_unit1(fine_conv2, coarse_conv2)
        fembed = self.fine_hemi.conv2[3](fembed)
        cembed = self.coarse_hemi.conv2[3](cembed)

        # 处理res1层
        fembed = self.fine_hemi.res1(fembed)
        cembed = self.coarse_hemi.res1(cembed)
        
        # 在conv3层后进行特征加权和
        fembed = self.fine_hemi.conv3(fembed)
        cembed = self.coarse_hemi.conv3(cembed)
        fine_conv3 = self.fine_features['fine_conv3']
        coarse_conv3 = self.coarse_features['coarse_conv3']
        fembed, cembed = self.gated_unit2(fine_conv3, coarse_conv3)
        fembed = self.fine_hemi.conv3[3](fembed)
        cembed = self.coarse_hemi.conv3[3](cembed)

        # 在conv4层进行处理
        fembed = self.fine_hemi.conv4(fembed)
        cembed = self.coarse_hemi.conv4(cembed)

        # 处理res2层
        fembed = self.fine_hemi.res2(fembed)
        cembed = self.coarse_hemi.res2(cembed)
        
        # 拼接最终的嵌入特征，保持空间结构
        embed = torch.cat([fembed, cembed], dim=1)
        
        if self.mode_out == 'feature':
            return embed
        elif self.mode_out == 'pred':
            f_out = self.fine_head(embed)
            c_out = self.coarse_head(embed)
            return f_out, c_out

    def pool_and_combine(self, fine_feature, coarse_feature):
        fine_pooled = F.adaptive_avg_pool2d(fine_feature, (fine_feature.size(2) // 2, fine_feature.size(3) // 2))
        coarse_pooled = F.adaptive_avg_pool2d(coarse_feature, (coarse_feature.size(2) // 2, coarse_feature.size(3) // 2))
        combined_feature = fine_pooled + coarse_pooled
        return combined_feature   

    # def pool_and_combine(self, fine_feature, coarse_feature, target_size):
    # # 自适应平均池化以减少特征图大小
    #     fine_pooled = F.adaptive_avg_pool2d(fine_feature, (fine_feature.size(2) // 2, fine_feature.size(3) // 2))
    #     coarse_pooled = F.adaptive_avg_pool2d(coarse_feature, (coarse_feature.size(2) // 2, coarse_feature.size(3) // 2))
        
    #     # 使用加法合并特征
    #     combined_feature = fine_pooled + coarse_pooled

    #     # 上采样以匹配目标层的尺寸
    #     combined_feature = F.interpolate(combined_feature, size=target_size, mode='nearest')

    #     return combined_feature

    def save_feature_hook(self, feature_dict, layer_name):
        def hook(module, input, output):
            # print(f"Hook triggered for {layer_name}: output shape {output.shape}")
            feature_dict[layer_name] = output
        return hook

class UnilateralNet(nn.Module):
    """
    A Unilateral network with one hemisphere and one or two heads.
    The hemisphere and head type(s) are configurable.
    The heads can be for fine, coarse or both labels respectively.
    """

    def __init__(self,
                 mode_heads: str,
                 mode_out: str,
                 arch: str,
                 model_path: str,
                 freeze_params: bool,
                 k: float, per_k: float,
                 dropout: float,
                 for_ensemble: bool = False):
        """ Initialize UnilateralNet

        Args:
            arch (str): architecture for the hemisphere
            mode_heads (str): which heads to create
                                both = create fine and coarse heads
                                fine, coarse = just fine or coarse
            mode_out (str): where to get the outputs from: pred (head output) or feature (hemisphere(s) output)
        """
        super(UnilateralNet, self).__init__()
        
        logger.debug(f"------- Initialize UnilateralNet with mode_heads: {mode_heads}, mode_out: {mode_out}, arch: {arch}, model_path: {model_path}, k: {k}, per_k: {per_k}, freeze_params: {freeze_params}, dropout: {dropout}")

        if for_ensemble and mode_heads != 'both':
            raise ValueError('For ensemble, mode_heads must be "both"')
        check_modes(mode_heads, mode_out)

        self.mode_heads = mode_heads
        self.mode_out = mode_out

        # create the hemispheres
        self.hemisphere = globals()[arch](Namespace(**{"k": k, "k_percent": per_k,}))
        
        # add heads
        num_features = self.hemisphere.num_features
        logger.debug(f"-- num_features: {num_features}")

        # if self.mode_heads == 'fine' or self.mode_heads == 'both':        
        #     self.fine_head = add_head(num_features, 100, dropout)
        # if self.mode_heads == 'coarse' or self.mode_heads == 'both':                    
        #     self.coarse_head = add_head(num_features, 20, dropout)

        if self.mode_heads == 'fine' or self.mode_heads == 'both':        
            self.fine_head = add_head(num_features, 10, dropout)
        if self.mode_heads == 'coarse' or self.mode_heads == 'both':                    
            self.coarse_head = add_head(num_features, 10, dropout)

        if model_path is not None and model_path != '':
            logger.debug("------- Load hemisphere")
            load_uni_model(self, model_path)

    def forward(self, x):
        embed = self.hemisphere(x)

        if self.mode_out == 'feature':
            return embed
        elif self.mode_out == 'pred':
            if self.mode_heads == 'fine':
                fh = self.fine_head(embed)
                return fh
            if self.mode_heads == 'coarse':
                ch = self.coarse_head(embed)
                return ch
            if self.mode_heads == 'both':
                fh = self.fine_head(embed)
                ch = self.coarse_head(embed)
                return fh, ch        


class EnsembleNet(nn.Module):
    """
    An Ensemble network with multiple UnilateralNet models.
    Assumes each one was created with two heads.
    Averages the classification outputs from all of them.
    Designed for EVALUATION ONLY of a set of trained models.
    """

    def __init__(self,
                 arch: str,
                 model_path_list: list,
                 freeze_params: bool,
                 k: float, per_k: float,
                 dropout: float):
        """ Initialize EnsembleNet

        Args:
            arch (str): architecture for the hemisphere
            mode (str): which heads to create and where to get output
                                both = create fine and coarse heads, get classification output
                                fine, coarse = just fine or coarse, get classification output
                                features = don't create heads, get output features as output
        """
        super(EnsembleNet, self).__init__()
        
        logger.debug(f"------- Initialize EnsembleNet with arch: {arch}, model_path_list: {model_path_list}, k: {k}, per_k: {per_k}, freeze_params: {freeze_params}, dropout: {dropout}")

        def create_load_model(model_path):
            model = UnilateralNet('both', 'pred', arch, model_path, freeze_params, k, per_k, dropout, for_ensemble=True)
            load_uni_model(model, model_path)
            return model

        self.model_list = nn.ModuleList([create_load_model(model_path) for model_path in model_path_list])

    def forward(self, x):
        outputs = []
        for model in self.model_list:
            outputs.append(model(x))

        avg_output_f = torch.mean(torch.stack([output[0] for output in outputs]), dim=0)
        avg_output_c = torch.mean(torch.stack([output[1] for output in outputs]), dim=0)

        return avg_output_f, avg_output_c

class AttentionWeightLayer(nn.Module):
    def __init__(self, channel_size):
        super(AttentionWeightLayer, self).__init__()
        self.weights = nn.Parameter(torch.rand(2))  # 初始化权重

    def forward(self, fine_feature, coarse_feature):
        # 权重归一化
        weights = torch.softmax(self.weights, dim=0)
        # 加权特征融合
        combined_feature = weights[0] * fine_feature + weights[1] * coarse_feature
        return combined_feature

class GatedIntercommunicationUnit(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(GatedIntercommunicationUnit, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # x 和 y 是来自两个不同分支的输入
        identity_x = x
        identity_y = y

        # 基于y生成门控信号调整x
        y_gate = self.avg_pool(y)
        y_gate = self.fc1(y_gate)
        y_gate = self.relu(y_gate)
        y_gate = self.fc2(y_gate)
        gate_y = self.sigmoid(y_gate)
        x_adjusted = identity_x * gate_y
        # x_adjusted = identity_x +  gate_y * identity_x 

        # 基于x生成门控信号调整y
        x_gate = self.avg_pool(x)
        x_gate = self.fc1(x_gate)
        x_gate = self.relu(x_gate)
        x_gate = self.fc2(x_gate)
        gate_x = self.sigmoid(x_gate)
        y_adjusted = identity_y * gate_x
        # y_adjusted = identity_y + gate_x * identity_y

        # alpha = 0.5  # 可调整的混合系数
        # y_adjusted = (1 - alpha) * identity_y + alpha * gate_x * identity_y

        # y_adjusted = identity_y * torch.sigmoid(gate_x)


        return x_adjusted, y_adjusted

class GatedIntercommunicationUnit2(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(GatedIntercommunicationUnit2, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        # 初始化处理单个输入的模块
        self.single_input_gate = SingleInputGate(channels, reduction_ratio)
    
    def forward(self, x, y):
        # 使用单个输入门控单元处理两个分支的输入
        adjusted_x = self.single_input_gate(x, y)
        adjusted_y = self.single_input_gate(y, x)
        return adjusted_x, adjusted_y

class SingleInputGate(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(SingleInputGate, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction_ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction_ratio, channels)
        self.sigmoid = nn.Sigmoid()
        
        self.conv1 = nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels // reduction_ratio)
        self.conv2 = nn.Conv2d(channels // reduction_ratio, channels // reduction_ratio, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels // reduction_ratio)
        self.conv3 = nn.Conv2d(channels // reduction_ratio, 1, 1, bias=False)
        self.relu_sp = nn.ReLU()

    def forward(self, main_input, aux_input):
        # Channel-wise from auxiliary input
        channel_gate = self.gap(aux_input) # C x 1 x 1
        channel_gate = channel_gate.view(channel_gate.size(0), -1) # Flatten
        channel_gate = self.fc1(channel_gate)
        channel_gate = self.relu(channel_gate)
        channel_gate = self.fc2(channel_gate)
        channel_gate = self.sigmoid(channel_gate)
        channel_gate = channel_gate.view(channel_gate.size(0), self.channels, 1, 1)
        
        # Spatial-wise from main input
        spatial_gate = self.conv1(main_input)
        spatial_gate = self.bn1(spatial_gate)
        spatial_gate = self.relu(spatial_gate)
        spatial_gate = self.conv2(spatial_gate)
        spatial_gate = self.bn2(spatial_gate)
        spatial_gate = self.relu(spatial_gate)
        spatial_gate = self.conv3(spatial_gate)
        spatial_gate = self.relu_sp(spatial_gate) # C x H x W
        
        # Apply gates on main input
        adjusted_main_input = main_input * channel_gate * spatial_gate
        return adjusted_main_input

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(query, key, value, attn_mask=mask)[0]
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_length=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_length, embed_size)
        for pos in range(max_length):
            for i in range(0, embed_size, 2):
                pe[pos, i] = torch.sin(pos / (10000 ** ((2 * i)/embed_size)))
                pe[pos, i + 1] = torch.cos(pos / (10000 ** ((2 * i)/embed_size)))
        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class FeatureTransformer(nn.Module):
    def __init__(self, embed_size, heads, depth, forward_expansion, dropout, max_length):
        super(FeatureTransformer, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_size, max_length)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion)
            for _ in range(depth)
        ])

    def forward(self, x, mask):
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, x, x, mask)
        return x
    
# def save_feature_hook(self, layer_name):
#         def hook(module, input, output):
#             self.__dict__[layer_name] = output
#         return hook

def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False

def freeze_params2(model, unfreeze_layers=None):
    if unfreeze_layers is None:
        unfreeze_layers = []
    for name, param in model.named_parameters():
        if not any([ul in name for ul in unfreeze_layers]):
            param.requires_grad = False
        else:
            param.requires_grad = True


def bilateral(args):
    """ Return a single hemisphere or bilateral (two hemispheres) network with two heads, based on the args
        See BilateralNet class for more details
    """
    return BilateralNet(args.mode_out,
                        args.farch, args.carch, 
                        args.fmodel_path, args.cmodel_path,
                        args.ffreeze_params, args.cfreeze_params,
                        args.fine_k, args.fine_per_k,
                        args.coarse_k ,args.coarse_per_k,
                        args.dropout,
                        args.cc_model, 
                        args.learning_type, 
                        args.connection_type, args.concat_method)

def unilateral(args):
    """ Return a unilateral network (one hemisphere) with one head, based on the args
        See UnilateralNet class for more details
    """
    return UnilateralNet(args.mode_heads,
                         args.mode_out,
                         args.farch,
                         args.fmodel_path, 
                         args.ffreeze_params,
                         args.fine_k, args.fine_per_k,
                         args.dropout)

def ensemble(args):
    return EnsembleNet(args.farch,
                       args.fmodel_path, 
                       args.ffreeze_params,
                       args.fine_k, args.fine_per_k,
                       args.dropout)

def load_uni_model(model, ckpt_path):
    """ 
        Load hemisphere and heads
        The hemisphere was trained with UnilateralNet, and it's being loaded again directly 
        We need to take out the namespace variables that don't apply to the model directly
    """
    sdict = torch.load(ckpt_path)['state_dict']
    model_dict = {k.replace('model.', ''):v for k,v in sdict.items()}
    model.load_state_dict(model_dict)
    return model

def load_hemi_model(model, ckpt_path):
    """ 
        Load hemisphere (not heads) 
        The hemisphere was trained with UnilateralNet, and it's being prepared for BilateralNet
        So we need to strip out the heads (with strict=False)
        We need to take out the namespace variables that don't apply to the model directly
    """
    sdict = torch.load(ckpt_path)['state_dict']
    model_dict = {k.replace('model.', '').replace('hemisphere.', ''):v for k,v in sdict.items()}
    model.load_state_dict(model_dict, strict=False)
    return model

############## THESE ONES USED BY GRADCAM AND TEST ##############

def load_model(model, ckpt_path):
    """[summary]

    Args:
        ckpt_path ([type]): [description]

    Returns:
        [type]: [description]
    """

    # TODO this version is used by gradcam, so will need to update with new names

    sdict = torch.load(ckpt_path)['state_dict']

    model_dict = {k.replace('model.', '').replace('encoder.', ''):v for k,v in sdict.items()}
    model.load_state_dict(model_dict)
    return model

def load_bicam_model(model, ckpt_path):
    """[summary]

    Args:
        ckpt_path ([type]): [description]

    Returns:
        [type]: [description]
    """

    # TODO I belive this is used to load the bilateral model. Both hemispheres and heads.
    sdict = torch.load(ckpt_path)['state_dict']
    model_dict = {k.replace('model_', '').replace('encoder.', ''):v for k,v in sdict.items() if not 'combiner' in k and not 'fc' in k}
    fc_dict = {k.replace('combiner.', '').replace('broad.', 'ccombiner.').replace('narrow.', 'fcombiner.'):v for k,v in sdict.items() if 'combiner' in k}
    model_dict.update(fc_dict)
    model.load_state_dict(model_dict)
    return model

def load_feat_model(model, ckpt_path):
    """[summary]

    Args:
        ckpt_path ([type]): [description]

    Returns:
        [type]: [description]
    """

    # TODO not sure what encoder is for ... fix when need to use this
    sdict = torch.load(ckpt_path)['state_dict']
    model_dict = {k.replace('model.', '').replace('encoder.', ''):v for k,v in sdict.items() if 'fc' not in k}
    model.load_state_dict(model_dict)
    return model
