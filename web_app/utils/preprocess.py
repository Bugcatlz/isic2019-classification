import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Must match dataloader ANATOM_SITES
ANATOM_SITES = [
    "anterior torso",
    "lower extremity",
    "upper extremity",
    "posterior torso",
    "lateral torso",
    "head/neck",
    "palm/soles",
    "oral/genital",
]

# Same as validation transform
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    # PIL image → (1, 3, 224, 224) tensor
    img = val_transform(pil_img)
    return img.unsqueeze(0)


def preprocess_metadata(age: int, sex: str, site: str) -> torch.Tensor:
    # Form fields → metadata vector
    meta = []

    # Age scaled /100
    age_val = float(age) if age is not None else 0.0
    meta.append(age_val / 100.0)

    # Sex mapping (missing = 0.5)
    if sex == "male":
        sex_val = 1.0
    elif sex == "female":
        sex_val = 0.0
    else:
        sex_val = 0.5
    meta.append(sex_val)

    # Anatomical site one-hot
    site_vec = [0.0] * len(ANATOM_SITES)
    if site in ANATOM_SITES:
        idx = ANATOM_SITES.index(site)
        site_vec[idx] = 1.0
    # Unknown stays all zeros
    meta.extend(site_vec)

    meta_np = np.array(meta, dtype=np.float32)
    return torch.from_numpy(meta_np)
