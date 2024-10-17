import torch
import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=100, dim=384, depth=6, heads=8, mlp_dim=768):
        super(ViT, self).__init__()
        assert image_size % patch_size == 0, "图像大小必须能被 patch 大小整除"

        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size  # CIFAR-100 是 RGB 图像

        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(0.1)

        # Transformer 编码器
        self.transformer = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                activation='gelu'
            ) for _ in range(depth)
        ])

        self.to_cls_token = nn.Identity()

        # 分类头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        p = 4
        x = img.reshape(img.shape[0], 3, self.image_size, self.image_size)
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(img.shape[0], -1, p*p*3)
        x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        x = self.mlp_head(x)
        return x

# 测试模型的参数量
model = ViT()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"参数量: {total_params / 1e6:.2f}M")
