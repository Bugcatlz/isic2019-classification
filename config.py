import torch

# Training parameters
batch_size = 128
num_epochs = 40
learning_rate = 4e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset paths
csv_file = "ISIC_2019/ISIC_2019_Training_GroundTruth.csv"
img_dir = "ISIC_2019/ISIC_2019_Training_Input"
