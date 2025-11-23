# ISIC 2019 Skin Lesion Classification

This is a **Data Science Project** for multi-class skin lesion classification using the ISIC 2019 dataset.  
The project applies deep learning models with custom data augmentation and Focal Loss to handle class imbalance.

---

## Project Structure
```
DSP/
├── dataloader.py         # Dataset class
├── transforms.py         # Data augmentation
├── losses.py             # Custom loss functions
├── models.py             # Model definition (CNN + Transformer + Metadata)
├── utils.py              # Training / evaluation utilities
├── config.py             # Hyperparameters & paths
├── train.py              # Main training script
├── requirements.txt      # All dependencies
├── splits/               # Train/Val/Test CSV splits (auto-generated)
├── checkpoints/          # Saved model checkpoints
├── web_app/              # Explainable AI web application
│   ├── main.py           # FastAPI application
│   ├── explainer/        # Layer-CAM, Attention, SHAP
│   └── templates/        # HTML interface
└── README.md             # This file
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

## How to Run

### Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

### Training

1. Train the model:

   ```bash
   python train.py
   ```

2. The script will:
   * Automatically create CSV splits in `splits/`
   * Save the best model in `checkpoints/best_model.pth`

### Web Application

1. Run the web server:

   ```bash
   uvicorn web_app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. Access at `http://localhost:8000`

3. Usage:
   * Upload dermoscopic image
   * Enter metadata (age, sex, anatomic site)
   * View prediction with Layer-CAM, Attention, and SHAP explanations

---

## Notes

* Training parameters (batch size, epochs, learning rate) can be modified in `config.py`.
* You can replace the backbone in `models.py` to experiment with different architectures.
* Web app auto-cleans generated images older than 1 hour.