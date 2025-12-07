import torch
from tqdm import tqdm
import pandas as pd


from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassAUROC,
)

# Training loop for one epoch
def train_one_epoch(model, loader, criterion, optimizer, device, num_classes=8):
    model.train()
    running_loss, total = 0.0, 0
    all_labels = []

    metrics = MetricCollection({
        "acc": Accuracy(task="multiclass", num_classes=num_classes),
        "precision": MulticlassPrecision(num_classes=num_classes, average="macro"),
        "recall": MulticlassRecall(num_classes=num_classes, average="macro"),
        "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
        "macro_auc": MulticlassAUROC(num_classes=num_classes, average="macro"),
        "acc_per_class": MulticlassAccuracy(num_classes=num_classes, average=None),
        "precision_per_class": MulticlassPrecision(num_classes=num_classes, average=None),
        "recall_per_class": MulticlassRecall(num_classes=num_classes, average=None),
        "f1_per_class": MulticlassF1Score(num_classes=num_classes, average=None),
        "auc_per_class": MulticlassAUROC(num_classes=num_classes, average=None),

    }).to(device)

    loop = tqdm(loader, desc="[Train]", leave=False)

    for i, (images, labels, metas) in enumerate(loop):
        labels = labels.argmax(dim=1).to(device, non_blocking=True)
        images = images.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)

        outputs = model(images, metas)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        all_labels.append(labels)

        probs = torch.softmax(outputs, dim=1)
        metrics.update(probs, labels)
        result = metrics.compute()

        if(i%10 == 0):
            loop.set_postfix(
                aaloss=loss.item(),
                acc=float(result["acc"]),
                pre=float(result["precision"]),
                rec=float(result["recall"]),
                f1=float(result["f1"]),
                auc=float(result["macro_auc"]),
            )
    all_labels_tensor = torch.cat(all_labels)
    class_counts = torch.bincount(all_labels_tensor, minlength=num_classes).tolist()

    result = metrics.compute()
    result["class_counts"] = class_counts

    return running_loss / total, result

# Evaluation loop (validation / test)
def evaluate(model, loader, criterion, device, mode="[Val]", num_classes=8):
    model.eval()
    running_loss, total = 0.0, 0
    all_labels = []

    num_classes = num_classes

    metrics = MetricCollection({
        "acc": Accuracy(task="multiclass", num_classes=num_classes),
        "precision": MulticlassPrecision(num_classes=num_classes, average="macro"),
        "recall": MulticlassRecall(num_classes=num_classes, average="macro"),
        "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
        "macro_auc": MulticlassAUROC(num_classes=num_classes, average="macro"),
        "acc_per_class": MulticlassAccuracy(num_classes=num_classes, average=None),
        "precision_per_class": MulticlassPrecision(num_classes=num_classes, average=None),
        "recall_per_class": MulticlassRecall(num_classes=num_classes, average=None),
        "f1_per_class": MulticlassF1Score(num_classes=num_classes, average=None),
        "auc_per_class": MulticlassAUROC(num_classes=num_classes, average=None),

    }).to(device)

    loop = tqdm(loader, desc=mode, leave=False)
    with torch.no_grad():
        for i, (images, labels, metas) in enumerate(loop):
            labels = labels.argmax(dim=1).to(device, non_blocking=True)
            images = images.to(device, non_blocking=True)
            metas = metas.to(device, non_blocking=True)

            outputs = model(images, metas)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            all_labels.append(labels)

            probs = torch.softmax(outputs, dim=1)
            metrics.update(probs, labels)
            result = metrics.compute()

            if(i%10 == 0):
                loop.set_postfix(
                    aaloss=loss.item(),
                    acc=float(result["acc"]),
                    pre=float(result["precision"]),
                    rec=float(result["recall"]),
                    f1=float(result["f1"]),
                    auc=float(result["macro_auc"]),
                )

        all_labels_tensor = torch.cat(all_labels)
        class_counts = torch.bincount(all_labels_tensor, minlength=num_classes).tolist()

        result = metrics.compute()
        result["class_counts"] = class_counts

    return running_loss / total, result

def print_result(result, classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]):
    data = {
        "Class": ["macro"] + classes,
        "Count": [sum(result["class_counts"])] + result["class_counts"],
        "Accuracy": [result['acc'].item()] + result['acc_per_class'].tolist(),
        "Precision": [result['precision'].item()] + result['precision_per_class'].tolist(),
        "Recall": [result['recall'].item()] + result['recall_per_class'].tolist(),
        "F1": [result['f1'].item()] + result['f1_per_class'].tolist(),
        "AUC": [result['macro_auc'].item()] + result['auc_per_class'].tolist(),
    }

    df = pd.DataFrame(data)
    print(df)
