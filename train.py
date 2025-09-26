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
from utils import train_one_epoch, evaluate
import config

# Create directories for splits and checkpoints if they do not exist
os.makedirs("splits", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Dataset split
df = pd.read_csv(config.csv_file)
df["label"] = df.iloc[:, 1:].idxmax(axis=1)

train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.3333, stratify=temp_df["label"], random_state=42)

train_df.to_csv("splits/train_split.csv", index=False)
val_df.to_csv("splits/val_split.csv", index=False)
test_df.to_csv("splits/test_split.csv", index=False)

# Dataset and loaders
train_dataset = ISIC2019Dataset(config.img_dir, "splits/train_split.csv", transform=train_transform)
val_dataset   = ISIC2019Dataset(config.img_dir, "splits/val_split.csv", transform=val_transform)
test_dataset  = ISIC2019Dataset(config.img_dir, "splits/test_split.csv", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Model, loss, optimizer
num_classes = len(train_dataset.label_cols)
model = build_model(num_classes).to(config.device)

criterion = FocalLoss(alpha=1, gamma=2)
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

# Training loop
best_acc = 0.0
for epoch in range(1, config.num_epochs + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, config.device, mode="[Val]")
    scheduler.step()

    print(f"Epoch [{epoch}] Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        save_path = os.path.join("checkpoints", "best_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved at {save_path} (Acc: {best_acc:.4f})")

# Test loop
print("\nTesting best model...")
model.load_state_dict(torch.load(os.path.join("checkpoints", "best_model.pth")))
test_loss, test_acc = evaluate(model, test_loader, criterion, config.device, mode="[Test]")
print(f"Test Accuracy: {test_acc:.4f}")
