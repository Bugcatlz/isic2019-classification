import torch
from tqdm import tqdm

# Training loop for one epoch
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    loop = tqdm(loader, desc="[Train]", leave=False)
    for images, labels in loop:
        labels = labels.argmax(dim=1).to(device, non_blocking=True)
        images = images.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

    return running_loss / total, correct / total

# Evaluation loop (validation / test)
def evaluate(model, loader, criterion, device, mode="[Val]"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    loop = tqdm(loader, desc=mode, leave=False)
    with torch.no_grad():
        for images, labels in loop:
            labels = labels.argmax(dim=1).to(device, non_blocking=True)
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

    return running_loss / total, correct / total
