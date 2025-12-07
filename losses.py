import torch
import torch.nn as nn

# Implementation of Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        # 处理 alpha
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        
        pt = torch.exp(-ce_loss)
        
        focal_term = (1 - pt) ** self.gamma
        
        loss = focal_term * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                
                alpha_t = self.alpha[targets]
                loss = alpha_t * loss
            else:
                loss = self.alpha * loss

        # 6. Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss