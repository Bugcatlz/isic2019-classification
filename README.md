# ISIC 2019 Skin Lesion Classification

This is a **Data Science Project** for multi-class skin lesion classification using the ISIC 2019 dataset.  
The project applies deep learning models with custom data augmentation and Focal Loss to handle class imbalance.

---

## Project Structure
```
ISIC2019_project/
│── dataloader.py         # Dataset class
│── transforms.py         # Data augmentation
│── losses.py             # Custom loss functions
│── models.py             # Model definition
│── utils.py              # Training / evaluation utilities
│── config.py             # Hyperparameters & paths
│── train.py              # Main training script
│── splits/               # Train/Val/Test CSV splits (generated automatically)
│── checkpoints/          # Saved model checkpoints
```
---

## Dataset
- **ISIC 2019 Skin Lesion Dataset**  
  Download from: [https://challenge2019.isic-archive.com/](https://challenge2019.isic-archive.com/)  
- After downloading, place the data in:
```
ISIC_2019/
├── ISIC_2019_Training_Input/          # Images
├── ISIC_2019_Training_GroundTruth.csv # Labels
````

> Note: The dataset is **not included** in this repository. You must download it manually.

---

##  How to Run
1. Train the model:

   ```bash
   python train.py
   ```

2. The script will:

   * Automatically create CSV splits in `splits/`
   * Save the best model in `checkpoints/best_model.pth`

---

## Notes

* Training parameters (batch size, epochs, learning rate) can be modified in `config.py`.
* You can replace the backbone in `models.py` to experiment with different architectures.