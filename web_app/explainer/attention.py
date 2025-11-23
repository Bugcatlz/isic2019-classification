import os
import uuid

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def generate_attention_rollout(attn_layers, original_image, base_dir):
    assert len(attn_layers) > 0, "No attention layers provided"

    # Step 1: convert all layers to head-averaged + normalized matrices
    processed = []

    for attn in attn_layers:
        A = attn[0]                 # batch=0 → (heads, N, N)
        A = A.mean(dim=0)           # → (N, N)
        N = A.size(0)

        I = torch.eye(N, device=A.device)
        A = A + I                   # add identity

        A = A / A.sum(dim=-1, keepdim=True)  # row normalize

        processed.append(A)

    # Step 2: rollout (matrix multiplication)
    A_rollout = processed[0]
    for i in range(1, len(processed)):
        A_rollout = processed[i] @ A_rollout   # Aᵢ × Aᵢ₋₁ × ... × A₁

    # Step 3: convert rollout to importance map
    token_importance = A_rollout.mean(dim=0)   # (N,)
    H = W = int(token_importance.numel() ** 0.5)
    attn_map = token_importance.reshape(H, W)

    # Normalize
    attn_map -= attn_map.min()
    if attn_map.max() > 0:
        attn_map /= attn_map.max()

    # Upsample to 224×224
    attn_map = attn_map.unsqueeze(0).unsqueeze(0)
    attn_map = F.interpolate(attn_map, size=(224, 224), mode="bilinear")
    attn_map = attn_map.squeeze().cpu().numpy()

    # Overlay
    orig_np = np.array(original_image.resize((224, 224)))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(orig_np)
    ax.imshow(attn_map, cmap="magma", alpha=0.4)
    ax.axis("off")

    results_dir = os.path.join(base_dir, "static", "results")
    os.makedirs(results_dir, exist_ok=True)
    filename = f"attn_rollout_{uuid.uuid4().hex}.png"
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return f"/static/results/{filename}"
