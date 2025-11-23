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

# Directories
os.makedirs("splits", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Load label CSV (multi-class one-hot)
print("[INFO] Loading label CSV:", config.csv_file)
df = pd.read_csv(config.csv_file)

# Create a single label column (string class name)
df["label"] = df.iloc[:, 1:].idxmax(axis=1)


# Load metadata CSV
print("[INFO] Loading metadata CSV:", config.metadata_file)
meta_df = pd.read_csv(config.metadata_file)

# Merge metadata with labels on image column
df = df.merge(meta_df, on="image", how="left")
print(f"[INFO] Dataset after merge: {df.shape}")


# Stratified split
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df["label"], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.3333, stratify=temp_df["label"], random_state=42
)

# Save split CSVs
train_df.to_csv("splits/train_split.csv", index=False)
val_df.to_csv("splits/val_split.csv", index=False)
test_df.to_csv("splits/test_split.csv", index=False)

print("[INFO] Splits saved → splits/train_split.csv, val_split.csv, test_split.csv")


# Datasets and loaders
train_dataset = ISIC2019Dataset(config.img_dir, "splits/train_split.csv", transform=train_transform)
val_dataset   = ISIC2019Dataset(config.img_dir, "splits/val_split.csv", transform=val_transform)
test_dataset  = ISIC2019Dataset(config.img_dir, "splits/test_split.csv", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=False)
val_loader   = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=False)
test_loader  = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True, persistent_workers=False)

print("[INFO] DataLoaders ready")


# Model, loss, optimizer
num_classes = len(train_dataset.label_cols)
model = build_model(num_classes).to(config.device)

criterion = FocalLoss(alpha=1, gamma=2)
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

print("[INFO] Model initialized")


# Training loop
best_acc = 0.0

for epoch in range(1, config.num_epochs + 1):
    print(f"\n========== Epoch {epoch}/{config.num_epochs} ==========")

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.device)
    val_loss, val_acc     = evaluate(model, val_loader, criterion, config.device, mode="[Val]")

    scheduler.step()

    print(f"[Epoch {epoch}] Train Acc: {train_acc:.4f}, Loss: {train_loss:.4f}")
    print(f"[Epoch {epoch}] Val   Acc: {val_acc:.4f}, Loss: {val_loss:.4f}")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        save_path = os.path.join("checkpoints", "best_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"[INFO] Best model saved → {save_path} (Acc: {best_acc:.4f})")


# Testing (best checkpoint)
print("\n[INFO] Testing best model...")
model.load_state_dict(torch.load("checkpoints/best_model.pth"))

test_loss, test_acc = evaluate(model, test_loader, criterion, config.device, mode="[Test]")

print(f"\n========== TEST RESULT ==========")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print("=================================")

# Cleanup DataLoaders to prevent resource leaks
del train_loader, val_loader, test_loader
import gc
gc.collect()
