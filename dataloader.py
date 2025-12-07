import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import config
from transforms import train_transform, val_transform
import random

class ISIC2019Dataset(Dataset):
    def __init__(self, img_dir, csv_file, meta_file=None, aug_dir = None, augmeta =  None, transform=None, train=False, undersampling = True):
        self.img_dir = img_dir
        self.aug_dir = aug_dir
        if(augmeta is not None):
            self.augmeta = pd.read_csv(augmeta)
        else:
            self.augmeta = []
        self.data = pd.read_csv(csv_file)
        self.meta = not(meta_file is None)
        if(self.meta):
            meta = pd.read_csv(meta_file)
            self.data = self.data.merge(meta, how='inner', on='image')
        features = ["age_approx", "anatom_site_general", "sex"]
        self.n_features = []
        self.features = {}
        for feature in features:
            tem = list(set(self.data[feature][self.data[feature].notna()]))
            tem.sort()
            self.n_features.append(len(tem)+1)
            tem = {tem[i]: i+1 for i in range(0,len(tem))}
            self.features[feature] = tem
        self.transform = transform
        self.train = train

        self.label_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

        # Remove rows with unknown labels
        if "UNK" in self.data.columns:
            self.data = self.data[self.data[self.label_cols].sum(axis=1) > 0].reset_index(drop=True)

        if (undersampling):
            majority = self.data[self.data['label'] == "NV"]
            other    = self.data[self.data['label'] != "NV"]
            minority = self.data[self.data['label'] == "MEL"]

            majority_under = majority.sample(n=len(minority), random_state=42)
            self.data = pd.concat([majority_under, other])
        self.real = len(self.data)
        if(not isinstance(self.augmeta, (list))):
            self.weights = [len(self.data[self.data['label'] == label]) + len(self.augmeta[self.augmeta['label'] == label])for label in self.label_cols]
            the_sum = sum(self.weights)
            self.weights = [the_sum /weight for weight in self.weights]
            the_mean = sum(self.weights) /len(self.weights)
            self.weights = [weight / the_mean for weight in self.weights]

        else:
            self.weights = [len(self.data[self.data['label'] == label]) for label in self.label_cols]

    def __len__(self):
        return self.real + len(self.augmeta)

    def __getitem__(self, idx):
        if(idx < self.real):
            row = self.data.iloc[idx]
            img_name = row["image"] + ".jpg"
            img_path = os.path.join(self.img_dir, img_name)
        else:
            row = self.augmeta.iloc[idx - self.real]
            img_name = f"generate_{(idx - self.real):05d}.png"
            img_path = os.path.join(self.aug_dir, img_name)

        # Load image in RGB mode
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        label = torch.tensor(row[self.label_cols].values.astype("float32"))
        if(not self.meta):
            return image, label
        meta = []
        for feature in list(self.features):
            if pd.isna(row[feature]):
                meta.append(0)
            elif(not self.train or random.random()>0.5):
                meta.append(self.features[feature][row[feature]])
            else:
                meta.append(0)
        meta = torch.tensor(meta, dtype=torch.long)
                
        return image, label, meta
        

if __name__ == "__main__":
    dataset = ISIC2019Dataset(config.img_dir, "splits/train_split.csv",
                               meta_file="data/ISIC_2019_Training_Metadata.csv",
                               aug_dir = config.augimg_dir,
                               augmeta = config.augmeta,
                                 transform=train_transform)
    train_loader = DataLoader(dataset, batch_size=config.batch_size,
                               shuffle=True, num_workers=2, pin_memory=True)
    print(f"generate_{500:05d}.png")
    