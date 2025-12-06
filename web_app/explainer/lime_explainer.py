import os
import uuid
import numpy as np
import torch
import torch.nn.functional as F
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_lime_explanation(model, img_tensor, meta_tensor, original_image, 
                              class_idx, device, base_dir, num_samples=1000):
    """
    Generate a scientifically rigorous LIME explanation for ISIC skin lesion analysis.
    Style: Standard Superpixel Boundaries (The "Yellow Cracks" style).
    Best for: Verifying model focus on lesion borders vs. background artifacts.
    """
    try:
        # 1. Prepare image
        img_np = np.array(original_image.resize((224, 224)))

        # 2. Define the prediction function
        def predict_fn(images):
            """
            LIME input: numpy array (N, H, W, 3) uint8
            Model output: numpy array (N, num_classes) probabilities
            """
            model.eval()
            with torch.no_grad():
                imgs_tensor = torch.from_numpy(images).float() / 255.0
                # (N, H, W, C) -> (N, C, H, W)
                imgs_tensor = imgs_tensor.permute(0, 3, 1, 2).to(device)

                mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                imgs_tensor = (imgs_tensor - mean) / std

                batch_size = imgs_tensor.size(0)
                meta_batch = meta_tensor.to(device)
                if meta_batch.dim() == 1:
                    meta_batch = meta_batch.unsqueeze(0)
                meta_batch = meta_batch.repeat(batch_size, 1)

                logits, _, _ = model(imgs_tensor, meta_batch)
                probs = torch.softmax(logits, dim=1)
                
                return probs.cpu().numpy()

        # 3. LIME Explainer
        explainer = lime_image.LimeImageExplainer()
        
        explanation = explainer.explain_instance(
            img_np, 
            predict_fn, 
            top_labels=1, 
            hide_color=0, 
            num_samples=num_samples,
            batch_size=32 
        )

        # 4. Get image and mask
        temp, mask = explanation.get_image_and_mask(
            class_idx,
            positive_only=True,
            num_features=1,  
            hide_rest=False 
        )

        # 5. Create boundary overlay
        img_boundry = mark_boundaries(temp / 255.0, mask, color=(1, 1, 0), mode='thick')

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img_boundry)
        ax.axis("off")

        # 6. Save the figure
        results_dir = os.path.join(base_dir, "static", "results")
        os.makedirs(results_dir, exist_ok=True)

        filename = f"lime_result_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(results_dir, filename)

        plt.savefig(filepath, bbox_inches="tight", pad_inches=0, dpi=150)
        plt.close(fig)

        return f"/static/results/{filename}"

    except Exception as e:
        print(f"LIME generation error: {e}")
        import traceback
        traceback.print_exc()
        return None