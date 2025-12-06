import os
import uuid

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def generate_layercam(model, img_tensor, meta_tensor, original_image, class_idx, device, base_dir, layers):
    model.eval()
    model.zero_grad()

    activations = {}
    gradients = {}

    # hooks: forward + backward
    def forward_hook(layer_name):
        def _hook(module, inp, out):
            # handle modules that return lists/tuples (e.g. timm feature extractors)
            if isinstance(out, (list, tuple)):
                target_out = out[-1]
            elif isinstance(out, dict):
                # pick last value
                target_out = list(out.values())[-1]
            else:
                target_out = out

            # store detached activation for later use
            try:
                activations[layer_name] = target_out.detach()
            except Exception:
                # fallback if not a tensor
                activations[layer_name] = target_out

            # register a hook on the tensor itself to capture gradients
            try:
                # only tensors have register_hook
                if hasattr(target_out, 'register_hook'):
                    def _capture_grad(grad, name=layer_name):
                        gradients[name] = grad.detach()
                    target_out.register_hook(_capture_grad)
            except Exception:
                pass

        return _hook

    # register forward hooks only (we capture grads via tensor hooks)
    hook_handles = []
    for i, layer in enumerate(layers):
        name = f"layer_{i}"
        try:
            h = layer.register_forward_hook(forward_hook(name))
            hook_handles.append(h)
        except Exception:
            pass

    # forward + backward
    img_tensor.requires_grad_(True)
    logits, _, _ = model(img_tensor.to(device), meta_tensor.to(device))
    target = logits[0, class_idx]
    target.backward()

    # remove hooks
    for h in hook_handles:
        h.remove()

    # ---- compute LayerCAM for each layer ----
    cam_total = None

    for key in activations.keys():
        A_full = activations[key]
        G_full = gradients.get(key, None)

        if G_full is None:
            # no gradient captured for this layer — skip
            continue

        # select batch index 0
        if A_full.dim() == 4:
            A = A_full[0]
        else:
            A = A_full

        if G_full.dim() == 4:
            G = G_full[0]
        else:
            G = G_full

        pos_grad = torch.relu(G)
        cam = torch.relu(A * pos_grad).sum(dim=0)  # (H, W)

        # normalize each layer's CAM
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        # ---- fix: resize BEFORE fusion ----
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)   # now (224,224)

        # accumulate
        if cam_total is None:
            cam_total = cam
        else:
            cam_total += cam

    # final normalize
    cam_total -= cam_total.min()
    if cam_total.max() > 0:
        cam_total /= cam_total.max()

    cam_np = cam_total.cpu().numpy()

    # overlay
    orig_np = np.array(original_image.resize((224,224)))
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(orig_np)
    ax.imshow(cam_np, cmap="magma", alpha=0.45)
    # ax.imshow(cam_np, cmap="jet", alpha=0.45)
    ax.axis("off")
    plt.tight_layout(pad=0)

    # save
    results_dir = os.path.join(base_dir, "static", "results")
    os.makedirs(results_dir, exist_ok=True)
    filename = f"layercam_{uuid.uuid4().hex}.png"
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return f"/static/results/{filename}"
