import numpy as np
import torch
import shap
from ..utils.preprocess import ANATOM_SITES

def generate_shap_metadata(
    model,
    image_tensor: torch.Tensor,
    meta_tensor: torch.Tensor,
    class_idx: int,
    device: str,
    top_k: int | None = 5,
    filter_zero: bool = True,
    zero_epsilon: float = 1e-8,
    nsamples: int = 100
):
    model.eval()

    # 1. Feature names Preparation
    meta_np = meta_tensor.detach().cpu().numpy()
    F_dim = meta_np.shape[1]
    
    feature_names = ["age_approx", "sex"] + [f"{s}" for s in ANATOM_SITES]

    if len(feature_names) > F_dim:
        feature_names = feature_names[:F_dim]
    elif len(feature_names) < F_dim:
        feature_names.extend([f"feat_{i}" for i in range(len(feature_names), F_dim)])

    # 2. Define prediction function (core logic)
    image_tensor = image_tensor.to(device)

    def predict_func(meta_batch_np: np.ndarray) -> np.ndarray:
        # meta_batch_np: (B, F)
        meta_batch = torch.from_numpy(meta_batch_np).float().to(device)
        B = meta_batch.size(0)

        img_batch = image_tensor.repeat(B, 1, 1, 1)

        with torch.no_grad():
            logits, _, _ = model(img_batch, meta_batch)
            probs = torch.softmax(logits, dim=1)
        
        return probs[:, class_idx:class_idx + 1].detach().cpu().numpy()

    # 3. Setup baseline (background) for SHAP
    background = np.zeros((1, F_dim), dtype=np.float32)
    background[0, 0] = 0.5 
    background[0, 1] = 0.5

    # 4. Run SHAP
    explainer = shap.KernelExplainer(predict_func, background)
    
    shap_values = explainer.shap_values(meta_np, nsamples=nsamples, silent=True)

    if isinstance(shap_values, list):
        shap_arr = shap_values[0]
    else:
        shap_arr = shap_values

    if len(shap_arr.shape) == 3:
        shap_arr = shap_arr.squeeze(-1)
    
    class_shap = shap_arr.reshape(-1)
    class_shap = np.asarray(class_shap, dtype=float)

    idx_sorted = np.argsort(np.abs(class_shap))[::-1]
    
    final_top_k = top_k if top_k is not None else len(idx_sorted)
    top_indices = idx_sorted[:final_top_k]

    results = []
    for i in top_indices:
        val = float(class_shap[i])
        if filter_zero and abs(val) < zero_epsilon:
            continue
            
        results.append({
            "name": str(feature_names[i]),
            "value": val,
        })

    if filter_zero and not results and len(idx_sorted) > 0:
        i = idx_sorted[0]
        results.append({
            "name": str(feature_names[i]),
            "value": float(class_shap[i])
        })

    return results