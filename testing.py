import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

H_W = 512
shape = (H_W, H_W)

class ISIC2019Dataset(Dataset):
    def __init__(self, img_dir, csv_file, meta_file = None, transform=None, train = False):
        # Image directory and CSV file
        self.img_dir = img_dir
        self.data = pd.read_csv(csv_file)
        self.meta = not (meta_file is None)
        if(self.meta):
            meta = pd.read_csv(meta_file)
            self.data = self.data.merge(meta, how='inner', on='image')
        self.transform = transform
        self.train = train

        # List of target classes
        self.label_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

        # Remove rows with unknown labels
        if "UNK" in self.data.columns:
            self.data = self.data[self.data[self.label_cols].sum(axis=1) > 0].reset_index(drop=True)
        print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row["image"] + ".jpg"
        img_path = os.path.join(self.img_dir, img_name)

        image = np.array(Image.open(img_path).resize(shape).convert("RGB"))[:,:,:3]/127.5 -1
        label = torch.tensor(row[self.label_cols].values.astype("float32"))


        return dict(jpg=image, txt = f"an patch of pathology image, graded {label} , including several cell")



if __name__ == "main":
    print("main")
    dataset = ISIC2019Dataset(img_dir="/data/Training",csv_file="/splits/train_split.csv")
    # dataloader = DataLoader(dataset, num_workers=4, batch_size=2, shuffle=True)
    # for i in dataloader:
    #     y = 0
