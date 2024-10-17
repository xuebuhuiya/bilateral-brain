import torch
import torch.nn as nn
from einops import rearrange

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x)  # [B_, N, 3C]
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B_, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个的形状为 [B_, num_heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.head_dim)))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)  # [B_, num_heads, N, head_dim]
        x = x.transpose(1, 2).reshape(B_, N, C)  # [B_, N, C]

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "输入特征的大小不匹配"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 对窗口进行划分
        x = x.unfold(1, self.window_size, self.window_size).unfold(2, self.window_size, self.window_size)
        x = x.contiguous().view(-1, self.window_size * self.window_size, C)

        # 自注意力
        x = self.attn(x)

        # 合并窗口
        x = x.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
        x = x.view(B, H * W, C)

        # 残差连接和 MLP
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class SwinTransformer(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=100, embed_dim=96, depths=[2, 2, 3], num_heads=[6, 12, 24]):
        super(SwinTransformer, self).__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim

        # Patch Embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.pos_drop = nn.Dropout(0.1)

        # Swin Transformer Layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList()
            for _ in range(depths[i_layer]):
                layer.append(
                    SwinTransformerBlock(
                        dim=int(embed_dim * 2 ** i_layer),
                        input_resolution=(image_size // patch_size // (2 ** i_layer), image_size // patch_size // (2 ** i_layer)),
                        num_heads=num_heads[i_layer],
                        window_size=7,
                        shift_size=0 if _ % 2 == 0 else 7 // 2
                    )
                )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))
        self.head = nn.Linear(int(embed_dim * 2 ** (self.num_layers - 1)), num_classes)

    def forward(self, x):
        x = self.patch_embed(x)  # [B, C, H, W]
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.pos_drop(x)

        for layer in self.layers:
            for block in layer:
                x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)  # 全局平均池化
        x = self.head(x)
        return x

# 测试模型的参数量
model = SwinTransformer()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"参数量: {total_params / 1e6:.2f}M")
