"""SHAP metadata contribution utilities."""

import numpy as np
import torch
import shap

from ..utils.preprocess import ANATOM_SITES


def generate_shap_metadata(
    model,
    image_tensor: torch.Tensor,  # (1, 3, H, W)
    meta_tensor: torch.Tensor,   # (1, F)
    class_idx: int,
    device: str,
    top_k: int | None = 5,
    filter_zero: bool = True,
    zero_epsilon: float = 1e-8,
):
    # Compute SHAP contributions for metadata features on target class probability

    model.eval()

    # Feature names
    meta_np = meta_tensor.detach().cpu().numpy()
    F_dim = meta_np.shape[1]

    # Order: age, sex, sites
    feature_names = ["age_approx", "sex"] + [f"{s}" for s in ANATOM_SITES]

    # Adjust length
    if len(feature_names) > F_dim:
        feature_names = feature_names[:F_dim]
    elif len(feature_names) < F_dim:
        for i in range(len(feature_names), F_dim):
            feature_names.append(f"feat_{i}")

    # Prediction function (returns target class probability)
    image_tensor = image_tensor.to(device)  # (1, 3, H, W)

    def predict_func(meta_batch_np: np.ndarray) -> np.ndarray:
        # meta_batch_np: (B, F) → (B, 1) target class probability
        meta_batch = torch.from_numpy(meta_batch_np).float().to(device)
        B = meta_batch.size(0)

        # Repeat image for metadata batch
        img_batch = image_tensor.repeat(B, 1, 1, 1)  # (B, 3, H, W)

        with torch.no_grad():
            logits, _, _ = model(img_batch, meta_batch)   # (B, C)
            probs = torch.softmax(logits, dim=1)          # (B, C)

        target_prob = probs[:, class_idx:class_idx + 1]   # (B, 1)
        return target_prob.detach().cpu().numpy()

    # Baseline (zeros)
    background = np.zeros((1, F_dim), dtype=np.float32)

    # SHAP explainer
    explainer = shap.KernelExplainer(predict_func, background)

    # Sample count (performance vs. accuracy)
    shap_values = explainer.shap_values(meta_np, nsamples=50)

    # Handle list vs ndarray
    if isinstance(shap_values, list):
        shap_arr = np.array(shap_values[0])  # (1, F)
    else:
        shap_arr = np.array(shap_values)     # expected (1, F)

    shap_arr = shap_arr.reshape(1, -1)
    class_shap = shap_arr[0]                # (F,)
    class_shap = np.asarray(class_shap, dtype=float)

    # Sort and select
    idx_sorted = np.argsort(np.abs(class_shap))[::-1]   # sort by absolute impact
    idx_sorted = [int(i) for i in idx_sorted]

    if top_k is None:
        top_indices = idx_sorted
    else:
        top_indices = idx_sorted[:top_k]

    results = []
    for i in top_indices:
        fname = str(feature_names[i])
        fval = float(class_shap[i])
        if filter_zero and abs(fval) < zero_epsilon:
            continue  # skip near-zero contribution
        results.append({
            "name": fname,
            "value": fval,
        })

    # If all filtered out, keep top 1 to avoid empty list
    if filter_zero and not results:
        i = idx_sorted[0]
        results.append({
            "name": str(feature_names[i]),
            "value": float(class_shap[i]),
        })

    return results
