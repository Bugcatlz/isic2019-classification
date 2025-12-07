import torch

# Training parameters
batch_size = 16
num_epochs = 200
learning_rate = 1e-4 * 1
momentum = 0.90
wweight_decay = 1e-5 * 1

warmup_epochs = 5
wlearning_rate = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset paths
csv_file = "data/ISIC_2019_Training_GroundTruth.csv"
img_dir = "data/Training"
augimg_dir = "data/Generate_200"
augmeta = "data/meta_generated.csv"