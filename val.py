import os
import glob
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
 
from models import build_model
from dataloader import ISIC2019Dataset
from transforms import val_transform
import config
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
import numpy as np
from tabulate import tabulate
 
 
def main():
    device = config.device
 
    # Prepare validation dataset/loader (use same splits and transforms as train.py)
    val_csv = os.path.join("splits", "val_split.csv")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Required split file not found: {val_csv}. Run `train.py` first to generate splits.")
 
    val_dataset = ISIC2019Dataset(config.img_dir, "splits/val_split.csv", meta_file="data/ISIC_2019_Training_Metadata.csv", transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)
 
    # Build model with correct number of classes
    num_classes = len(val_dataset.label_cols)
    model = build_model(model = "conxvit", num_classes=num_classes).to(device)
 
    # Find latest checkpoint
    checkpoint_dir = "checkpoints"
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
 
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    print(f"Loading checkpoint: {latest_checkpoint}")
 
    # Load state dict (support either raw state_dict or a dict wrapping it)
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
 
    model.load_state_dict(state)
 
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
 
    # Run evaluation loop to collect predictions and probabilities
    model.eval()
    all_probs = []
    all_preds = []
    all_trues = []
    total_loss = 0.0
    total_samples = 0
 
    with torch.no_grad():
        for images, metas, labels in tqdm(val_loader, desc="[Val]"):
            images = images.to(device)
            metas = metas.to(device)
            labels = labels.to(device)
 
            outputs, _, _ = model(images, metas)
            loss = criterion(outputs, labels.argmax(dim=1))
 
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
 
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
 
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_trues.append(labels.argmax(dim=1).cpu().numpy())
 
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
 
    val_loss = total_loss / max(1, total_samples)
    overall_acc = (all_preds == all_trues).mean()
 
    # Per-class precision/recall/f1
    precision, recall, f1, support = precision_recall_fscore_support(
        all_trues, all_preds, average=None, zero_division=0
    )
 
    # Per-class accuracy: correct / total per class
    per_class_acc = []
    for i in range(num_classes):
        idx = (all_trues == i)
        if idx.sum() == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append((all_preds[idx] == all_trues[idx]).sum() / idx.sum())
    per_class_acc = np.array(per_class_acc)
 
    # AUC per class (one-vs-rest)
    aucs = np.zeros(num_classes)
    try:
        # roc_auc_score requires one-hot true labels
        y_true_onehot = np.eye(num_classes)[all_trues]
        aucs = roc_auc_score(y_true_onehot, all_probs, average=None)
    except Exception:
        # If AUC cannot be computed for some classes (e.g., single-label present), set to nan
        aucs = np.array([np.nan] * num_classes)
 
    # Overall precision/recall/f1 (macro/micro/weighted)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_trues, all_preds, average='macro', zero_division=0
    )
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        all_trues, all_preds, average='micro', zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
        all_trues, all_preds, average='weighted', zero_division=0
    )
 
    overall_acc = accuracy_score(all_trues, all_preds)
 
    # AUC overall (macro / micro) if possible
    auc_macro, auc_micro = np.nan, np.nan
    try:
        y_true_onehot = np.eye(num_classes)[all_trues]
        auc_macro = roc_auc_score(y_true_onehot, all_probs, average='macro')
        auc_micro = roc_auc_score(y_true_onehot, all_probs, average='micro')
    except Exception:
        pass
 
    # Print results (tabulated if possible)
    print("\nValidation Results:")
    print(f"Loss: {val_loss:.4f}")
    print(f"Overall Accuracy: {100. * overall_acc:.2f}%")
 
    # Build per-class table
    per_class_rows = []
    for i, cls in enumerate(val_dataset.label_cols):
        p = precision[i]
        r = recall[i]
        f = f1[i]
        a = per_class_acc[i]
        auc = aucs[i]
        s = int(support[i])
        per_class_rows.append([cls, p, r, f, a, auc, s])
 
    headers = ["Class", "Precision", "Recall", "F1", "Acc", "AUC", "Support"]
 
    print("\nPer-class metrics:")
    print(tabulate(per_class_rows, headers=headers, tablefmt="github", floatfmt=".4f"))
 
    # Overall macro-only row
    overall_row = [["ALL-macro", p_macro, r_macro, f1_macro, overall_acc, auc_macro]]
    overall_headers = ["Metric", "Precision", "Recall", "F1", "Acc", "AUC_macro"]
    print("\nOverall metrics:")
    print(tabulate(overall_row, headers=overall_headers, tablefmt="github", floatfmt=".4f"))
 
 
if __name__ == '__main__':
    main()
 