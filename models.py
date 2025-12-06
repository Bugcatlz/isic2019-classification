import torch
import torch.nn as nn
import timm


# ============================================================
# Metadata Encoder
# ============================================================
class MetadataEncoder(nn.Module):
    def __init__(self, in_dim, embed_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, meta):
        return self.encoder(meta)  # [B, 64]


# ============================================================
# Hybrid Model: ConvNeXt-Tiny + ViT-Tiny + Metadata
# ============================================================
class SkinCancerConvNeXtViT(nn.Module):
    def __init__(self, num_classes=8, meta_dim=10, pretrained=True):
        super().__init__()

        # ====================================================
        # ConvNeXt Backbone
        # ====================================================
        self.convnext = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            features_only=True
        )
        self.convnext_out_dim = self.convnext.feature_info[-1]["num_chs"]  # usually 768

        # ====================================================
        # ViT Backbone
        # ====================================================
        self.vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=pretrained
        )

        # Dynamically detect CLS token dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = self.vit.forward_features(dummy)

            if isinstance(feats, dict):
                if "cls_token" in feats:
                    vit_dim = feats["cls_token"].shape[-1]
                elif "x" in feats:
                    vit_dim = feats["x"].shape[-1]
                else:
                    vit_dim = next(iter(feats.values())).shape[-1]
            else:
                vit_dim = feats.shape[-1]

        self.vit_dim = vit_dim

        # ====================================================
        # Metadata Encoder
        # ====================================================
        self.meta_encoder = MetadataEncoder(meta_dim, embed_dim=64)
        self.meta_dim = 64

        # ====================================================
        # Classifier Head
        # (concatenate raw convnext + vit + metadata features)
        # ====================================================
        fusion_dim = self.convnext_out_dim + self.vit_dim + self.meta_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    # ============================================================
    # Extract ViT CLS token
    # ============================================================
    def _forward_vit_cls(self, img):
        feats = self.vit.forward_features(img)

        if isinstance(feats, dict):
            if "cls_token" in feats:
                return feats["cls_token"][:, 0]
            elif "x" in feats:
                return feats["x"][:, 0]
            else:
                return next(iter(feats.values()))[:, 0]
        else:
            return feats[:, 0]

    # ============================================================
    # Forward Pass
    # ============================================================
    def forward(self, img, meta):

        # ConvNeXt features (GAP)
        conv_feat = self.convnext(img)[-1]
        conv_feat = conv_feat.mean(dim=[2, 3])  # [B, conv_dim]

        # ViT features (CLS)
        vit_feat = self._forward_vit_cls(img)    # [B, vit_dim]

        # Metadata features
        meta_feat = self.meta_encoder(meta)      # [B, 64]

        # Fusion
        fused = torch.cat([conv_feat, vit_feat, meta_feat], dim=1)

        # Classifier
        out = self.classifier(fused)

        return out, None, None


# ============================================================
# Factory Function
# ============================================================
def build_model(num_classes=8, meta_dim=10, pretrained=True):
    return SkinCancerConvNeXtViT(
        num_classes=num_classes,
        meta_dim=meta_dim,
        pretrained=pretrained
    )
