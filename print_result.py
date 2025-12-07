import pandas as pd
import torch

# 假設 result 是你 train_one_epoch 回傳的
# 這裡隨便先做個範例 tensor
num_classes = 4
result = {
    'acc': torch.tensor(0.85),
    'precision': torch.tensor(0.82),
    'recall': torch.tensor(0.80),
    'f1': torch.tensor(0.81),
    'macro_auc': torch.tensor(0.88),

    'acc_per_class': torch.tensor([0.9, 0.8, 0.85, 0.82]),
    'precision_per_class': torch.tensor([0.91, 0.78, 0.83, 0.78]),
    'recall_per_class': torch.tensor([0.88, 0.82, 0.84, 0.77]),
    'f1_per_class': torch.tensor([0.89, 0.80, 0.835, 0.775]),
    'auc_per_class': torch.tensor([0.95, 0.85, 0.88, 0.90]),
}

# 先整理成 dict of lists
data = {
    "Class": ["macro"] + [f"class_{i}" for i in range(num_classes)],
    "Accuracy": [result['acc'].item()] + result['acc_per_class'].tolist(),
    "Precision": [result['precision'].item()] + result['precision_per_class'].tolist(),
    "Recall": [result['recall'].item()] + result['recall_per_class'].tolist(),
    "F1": [result['f1'].item()] + result['f1_per_class'].tolist(),
    "AUC": [result['macro_auc'].item()] + result['auc_per_class'].tolist(),
}

df = pd.DataFrame(data)
print(df)
