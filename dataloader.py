import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

ANATOM_SITES = [
    "anterior torso", "lower extremity", "upper extremity", "posterior torso",
    "lateral torso", "head/neck", "palm/soles", "oral/genital"
]

class ISIC2019Dataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # Label columns
        self.label_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
        # Metadata columns
        self.meta_cols = ["age_approx", "sex", "anatom_site_general"]
        # Preprocess metadata
        self.data = self._preprocess_metadata(self.data)

    def _preprocess_metadata(self, df):  # returns df with engineered features
        # age_approx
        df["age_approx"] = df["age_approx"].fillna(0)
        # sex mapping
        df["sex"] = df["sex"].map({"male": 1, "female": 0})
        df["sex"] = df["sex"].fillna(0.5)  # missing = 0.5
        # anatom_site one-hot
        for site in ANATOM_SITES:
            df[f"site_{site}"] = (df["anatom_site_general"] == site).astype(float)

        df[ANATOM_SITES] = df[[f"site_{s}" for s in ANATOM_SITES]]

        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Image
        image_path = os.path.join(self.img_dir, row["image"] + ".jpg")
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Labels
        label = torch.tensor(row[self.label_cols].values.astype("float32"))
        # Metadata tensor
        meta = []

        # Age (normalize 0–100 → 0–1)
        meta.append(row["age_approx"] / 100)
        # Sex
        meta.append(row["sex"])
        # Anatom_site one-hot
        meta.extend(row[[f"site_{s}" for s in ANATOM_SITES]].values.tolist())

        meta = torch.tensor(meta).float()

        return image, meta, label
