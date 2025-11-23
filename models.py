import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN block
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Transformer block (returns attention weights)
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):  # x: (B, N, C)
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True)
        x = self.norm1(x + attn_out)

        x = self.norm2(x + self.mlp(x))

        return x, attn_weights

# Metadata encoder
class MetadataEncoder(nn.Module):
    def __init__(self, in_dim, embed_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.ReLU(),
        )

    def forward(self, meta):
        return self.encoder(meta)

# Hybrid model: CNN + Transformer + metadata fusion
class SkinCancerHybrid(nn.Module):
    def __init__(self, num_classes=8, meta_dim=10):
        super().__init__()

        # CNN feature extractor
        self.cnn1 = CNNBlock(3, 32)
        self.cnn2 = CNNBlock(32, 64)
        self.pool = nn.MaxPool2d(2, 2)
        # Patch embedding: stride 4 reduces 56×56 → 14×14 (tokens=196)
        self.patch_embed = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=4,
            stride=4
        )
        # Transformer blocks
        self.trans1 = TransformerBlock(dim=128)
        self.trans2 = TransformerBlock(dim=128)
        # Metadata encoder (1 age + 1 sex + 8 site = 10)
        self.metadata_encoder = MetadataEncoder(in_dim=meta_dim, embed_dim=64)
        # Classifier (fusion)
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, img, meta):
        # CNN features
        x = self.cnn1(img)
        x = self.pool(x)

        x = self.cnn2(x)
        x = self.pool(x)
        # Patch embedding
        x = self.patch_embed(x)  # (B, 128, 14, 14)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, 196, 128)
        # Transformer (global features)
        x, attn1 = self.trans1(x)
        x, attn2 = self.trans2(x)

        img_feature = x.mean(dim=1)  # (B, 128)
        # Metadata branch
        meta_feature = self.metadata_encoder(meta)  # (B, 64)
        # Fusion
        fused = torch.cat([img_feature, meta_feature], dim=1)
        # Prediction
        out = self.classifier(fused)
        # Return attention maps
        return out, attn1, attn2

# Factory
def build_model(num_classes, meta_dim=10):
    return SkinCancerHybrid(num_classes=num_classes, meta_dim=meta_dim)
