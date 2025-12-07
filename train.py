import os
import torch
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataloader import ISIC2019Dataset
from transforms import train_transform, val_transform
from models import build_model
from losses import FocalLoss
from utils import train_one_epoch, evaluate, print_result
import config

#Val  : Accuracy: 0.8859, Precision: 0.8612, Recall: 0.7790, F1: 0.8141, Loss: 0.4464 step 20 
#Val  : Accuracy: 0.8745, Precision: 0.8597, Recall: 0.7670, F1: 0.8044, Loss: 0.4716 step 200
#Val  : Accuracy: 0.8851, Precision: 0.8644, Recall: 0.7670, F1: 0.8066, Loss: 0.4261 ori
# Create directories for splits and checkpoints if they do not exist
os.makedirs("splits", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Dataset split
df = pd.read_csv(config.csv_file)
df["label"] = df.iloc[:, 1:].idxmax(axis=1)

train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.6666, stratify=temp_df["label"], random_state=42)

train_df.to_csv("splits/train_split.csv", index=False)
val_df.to_csv("splits/val_split.csv", index=False)
test_df.to_csv("splits/test_split.csv", index=False)

# Dataset and loaders
train_dataset = ISIC2019Dataset(config.img_dir, "splits/train_split.csv",
                                meta_file="data/ISIC_2019_Training_Metadata.csv",
                                # aug_dir = config.augimg_dir,
                                # augmeta = config.augmeta,
                                transform=train_transform,train=True, undersampling=False)
val_dataset   = ISIC2019Dataset(config.img_dir, "splits/val_split.csv",
                                meta_file="data/ISIC_2019_Training_Metadata.csv",
                                transform=val_transform, undersampling=False)
test_dataset  = ISIC2019Dataset(config.img_dir, "splits/test_split.csv",
                                meta_file="data/ISIC_2019_Training_Metadata.csv",
                                transform=val_transform, undersampling=False)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# Model, loss, optimizer
num_classes = len(train_dataset.label_cols)
model = build_model(num_classes, model="conxvit", meta = [] + train_dataset.n_features ).to(config.device)#

criterion = FocalLoss(alpha=train_dataset.weights, gamma=2)
# optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.wweight_decay)
# optimizer = optim.SGD(model.parameters(), momentum=config.momentum, lr=config.wlearning_rate, weight_decay=config.wweight_decay)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

# warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
#     optimizer, start_factor=config.wlearning_rate, end_factor=config.learning_rate, total_iters=config.warmup_epochs
# )

optimizer = torch.optim.SGD(
    model.parameters(),lr=config.learning_rate, momentum=config.momentum,weight_decay=config.wweight_decay
)

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,start_factor=1e-4, end_factor=1.0,total_iters=config.warmup_epochs
)

cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config.num_epochs - config.warmup_epochs
)

if __name__ == "__main__":
    # Training loop
    best_acc = 0.0
    for epoch in range(1, config.num_epochs + 1):
        t_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, config.device)
        v_loss, val___metrics = evaluate(model, val_loader, criterion, config.device, mode="[Val]")

        if(epoch <= config.warmup_epochs):
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        # t_acc = train_metrics["acc"]
        v_acc = val___metrics["acc"]
        # t_pre = train_metrics["precision"]
        # v_pre = val___metrics["precision"]
        # t_rec = train_metrics["recall"]
        # v_rec = val___metrics["recall"]
        # t_f1  = train_metrics["f1"]
        # v_f1  = val___metrics["f1"]

        print(f"Epoch [{epoch}] :")
        print_result(train_metrics)
        print_result(val___metrics)

        if v_acc > best_acc:
            best_acc = v_acc
            save_path = os.path.join("checkpoints", "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at {save_path} (Acc: {best_acc:.4f})")

    # Test loop
    print("\nTesting best model...")
    model.load_state_dict(torch.load(os.path.join("checkpoints", "best_model.pth")))
    test_loss, test_metrics = evaluate(model, test_loader, criterion, config.device, mode="[Test]")
    print_result(test_metrics)
