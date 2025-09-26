import torch.nn as nn
from torchvision import models

# Build model with custom classifier
def build_model(num_classes):
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model
