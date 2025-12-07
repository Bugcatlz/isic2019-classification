import torch
import torch.nn as nn
from torchvision import models


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.permute(*self.dims)
    
class MetaEmbedding(nn.Module):
    def __init__(self, meta, cls_dim=768 , gatted = True):
        super().__init__()
        emb_dims = cls_dim//4
        emb_list = [nn.Embedding(num, emb_dims, padding_idx=0) for num in meta]
        self.embeddings = nn.ModuleList(emb_list)

        total_dim = cls_dim + emb_dims*3

        self.mlp = nn.Sequential(
            nn.Linear(total_dim, cls_dim),
            nn.GELU(),
            nn.Linear(cls_dim, cls_dim),
        )

        self.gate = nn.Sequential(
            nn.Linear(total_dim, cls_dim),
            nn.Sigmoid()
        )

        self.gatted = gatted

    def forward(self, cls_token, meta_ids):
        if isinstance(meta_ids, torch.Tensor):
            meta_ids = [meta_ids[:, i] for i in range(meta_ids.shape[1])]

        meta_vecs = [emb(ids) for emb, ids in zip(self.embeddings, meta_ids)]

        x = torch.cat([cls_token] + meta_vecs, dim=-1)
        meta = self.mlp(x)

        if(self.gatted):
            meta = meta*self.gate(x)
        
        return meta

class Covxvit(nn.Module):
    def __init__(self, cnn, vit, meta):
        super().__init__()
        self.cnn = cnn
        self.vit = vit
        self.embadding = None
        if len(meta) > 0:
            self.embadding = MetaEmbedding(meta=meta, cls_dim=vit.class_token.shape[2])
        in_channels = cnn[6][1].out_channels
        self.proj = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(in_channels, vit.class_token.shape[2], kernel_size=1),
        )

    def forward(self, x, meta=None):
        x = self.cnn(x)
        x = self.proj(x)
        n, c, h, w = x.shape
        x = x.reshape(n, c, h*w)
        x = x.permute(0, 2, 1)
        n = x.shape[0]

        batch_class_token = self.vit.class_token.expand(n, -1, -1)

        if(not self.embadding is None):
            emb = self.embadding(batch_class_token[:,0,:], meta).unsqueeze(1)
            batch_class_token = batch_class_token + emb

        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit.encoder(x)
        x = x[:, 0]
        x = self.vit.heads(x)

        return x

def build_model(num_classes, model = "convnext", meta = []):
    if(model == "convnext"):
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif(model == "vitb"):
        model = models.vit_b_16(weights = models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif(model == "vit"):
        model = models.vit_l_16(weights = models.ViT_L_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif(model == "swinb"):
        model = models.swin_b(weights = models.Swin_B_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif(model == "swin"):
        model = models.swin_t(weights = models.Swin_T_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        # vit = models.vit_b_16(weights = models.ViT_B_16_Weights.IMAGENET1K_V1)
        vit = models.vit_l_32(weights = models.ViT_L_32_Weights.IMAGENET1K_V1)
        vit.heads.head = nn.Linear(vit.heads.head.in_features, num_classes)
        cnn = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
        model = Covxvit(cnn.features[:7], vit, meta)
    return model

if __name__ == "__main__":
    # model = build_model(10, model="conxvit", meta = [])
    
    # vit = models.vit_h_14(weights = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1) #weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    vit = models.vit_b_16()
    print(vit.class_token.shape)
    print(vit.encoder(torch.zeros((2,197,768))).shape)
    print(vit.conv_proj.out_channels)
    # print(nn.Conv2d(vit.conv_proj.out_channels, vit.class_token.shape[2], kernel_size=(1,1)))
    # vit = models.vit_b_32()  #768
    # print(vit.class_token.shape)
    # print(vit.encoder(torch.zeros((2,50,768))).shape)
    # vit = models.vit_l_16()
    # print(vit.class_token.shape)
    # print(vit.encoder(torch.zeros((2,197,1024))).shape)
    # vit = models.vit_l_32()  #1024
    # print(vit.class_token.shape)
    # print(vit.encoder(torch.zeros((2,50,1024))).shape)
    model = models.convnext_tiny()
    # print(model.features[:5](torch.zeros((1,3,224,224))).shape)
    # in_channels = model.features[:5][4][1].out_channels
    print(model.features[:5])
    # print(nn.Sequential(nn.LayerNorm((in_channels,), eps=1e-06, elementwise_affine=True), nn.Conv2d(in_channels, vit.class_token.shape[2], kernel_size=(1,1))))
    # model = models.convnext_small()
    # print(model.features[:5](torch.zeros((1,3,224,224))).shape)
    # model = models.convnext_base()
    # print(model.features[:5](torch.zeros((1,3,224,224))).shape)
    # model = models.convnext_large()
    # print(model.features[:5](torch.zeros((1,3,224,224))).shape)

