import os
import uuid
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def generate_attention_rollout(attn_layers, original_image, base_dir):
    """
    Generate Attention Rollout visualization for ViT.
    Fixes: Handles CLS token correctly and focuses on CLS attention.
    """
    assert len(attn_layers) > 0, "No attention layers provided"

    # Step 1: Preprocess matrices (Mean heads + Add Identity)
    processed = []
    for attn in attn_layers:
        # attn shape: (batch, heads, N, N) -> take batch 0
        A = attn[0]  
        
        # Average over heads
        A = A.mean(dim=0)  # (N, N)

        # To account for residual connections, we add an identity matrix
        # allowing attention to flow through unchanged.
        I = torch.eye(A.size(0), device=A.device)
        A = A + I
        
        # Renormalize the rows
        A = A / A.sum(dim=-1, keepdim=True)
        
        processed.append(A)

    # Step 2: Rollout (Matrix Multiplication)
    # Attention Rollout paper logic: Joint_Attn = A_layer @ Joint_Attn_prev
    joint_attention = processed[0]
    for i in range(1, len(processed)):
        joint_attention = processed[i] @ joint_attention

    # Step 3: Extract the specific attention from CLS token to image patches
    # joint_attention[0, 1:] 代表 CLS token 對後面所有 image patches 的關注度
    mask = joint_attention[0, 1:] 

    num_patches = mask.numel()
    grid_size = int(num_patches ** 0.5)
    
    if grid_size * grid_size != num_patches:
        print(f"Warning: Patch count {num_patches} is not a perfect square. Check if CLS token handling is correct.")
        width = int(np.ceil(num_patches**0.5))
        mask = F.pad(mask, (0, width*width - num_patches))
        grid_size = width

    attn_map = mask.reshape(grid_size, grid_size)

    # Normalize
    attn_map -= attn_map.min()
    if attn_map.max() > 0:
        attn_map /= attn_map.max()

    # Upsample
    attn_map = attn_map.unsqueeze(0).unsqueeze(0)
    attn_map = F.interpolate(attn_map, size=(224, 224), mode="bilinear")
    attn_map = attn_map.squeeze().cpu().numpy()

    # Overlay
    orig_np = np.array(original_image.resize((224, 224)))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(orig_np)
    ax.imshow(attn_map, cmap="inferno", alpha=0.5) 
    ax.axis("off")

    results_dir = os.path.join(base_dir, "static", "results")
    os.makedirs(results_dir, exist_ok=True)
    filename = f"attn_rollout_{uuid.uuid4().hex}.png"
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return f"/static/results/{filename}"