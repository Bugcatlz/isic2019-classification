import os
import uuid
import glob
from io import BytesIO
from typing import List
import io
import traceback
import time
from datetime import datetime

import torch
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from models import build_model  # Hybrid model
import config
from .utils.preprocess import preprocess_image, preprocess_metadata
from .explainer.layercam import generate_layercam
from .explainer.attention import generate_attention_rollout
from .explainer.shap_meta import generate_shap_metadata
from .explainer.lime_explainer import generate_lime_explanation


# App setup
app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(os.path.join(static_dir, "results"), exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Class names (label order)
CLASS_NAMES: List[str] = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# Disease information
DISEASE_INFO = {
    "MEL": {
        "name": "Melanoma",
        "description": "A serious form of skin cancer that develops in melanocytes. Early detection is crucial for successful treatment. Characterized by asymmetry, irregular borders, and color variation."
    },
    "NV": {
        "name": "Melanocytic Nevus",
        "description": "A benign mole or birthmark composed of melanocytes. Usually harmless but should be monitored for changes in size, shape, or color over time."
    },
    "BCC": {
        "name": "Basal Cell Carcinoma",
        "description": "The most common type of skin cancer, arising from basal cells. Grows slowly and rarely spreads, but can cause local damage if untreated. Often appears as a pearly or waxy bump."
    },
    "AK": {
        "name": "Actinic Keratosis",
        "description": "A precancerous skin lesion caused by sun damage. Appears as rough, scaly patches. Can potentially develop into squamous cell carcinoma if left untreated."
    },
    "BKL": {
        "name": "Benign Keratosis",
        "description": "A non-cancerous skin growth including seborrheic keratoses. Common in older adults, appearing as brown, black, or tan growths. Generally harmless and doesn't require treatment."
    },
    "DF": {
        "name": "Dermatofibroma",
        "description": "A benign fibrous nodule commonly found on the legs. Feels like a hard bump under the skin. Usually harmless and doesn't require treatment unless bothersome."
    },
    "VASC": {
        "name": "Vascular Lesion",
        "description": "Abnormalities of blood vessels in the skin, including hemangiomas and angiokeratomas. Most are benign and may appear as red or purple marks."
    },
    "SCC": {
        "name": "Squamous Cell Carcinoma",
        "description": "The second most common skin cancer, arising from squamous cells. Can spread if untreated. Often appears as a firm red nodule or flat lesion with a scaly surface."
    }
}

# Load model
MODEL_PATH = os.path.join(PARENT_DIR, "checkpoints", "best_model.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"

# meta_dim = 1(age) + 1(sex) + 8(anatom_site)
model = build_model(num_classes=len(CLASS_NAMES), meta_dim=10).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Model info
MODEL_INFO = {
    "name": "Hybrid ConvNeXt-Tiny + ViT-Tiny",
    "version": "1.0.0",
    "accuracy": "85.2%",  # 更新為你的實際準確率
    "dataset": "ISIC 2019",
    "classes": 8,
    "last_updated": "2024-12-04"
}


def cleanup_old_results(results_dir, max_age_seconds=3600):
    """Remove result images older than max_age_seconds"""
    try:
        current_time = time.time()
        for filepath in glob.glob(os.path.join(results_dir, "*.*")):
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > max_age_seconds:
                    os.remove(filepath)
    except Exception as e:
        print(f"Cleanup error: {e}")


def create_image_report(original_path, lime_path, layercam_path, attention_path, 
                       prediction, probability, age, sex, site, 
                       all_probs, shap_contribs, disease_info):
    """Create a comprehensive image report with web-style design"""
    try:
        # Canvas size
        width = 1400
        height = 1800
        
        # Create canvas with dark background
        canvas = Image.new('RGB', (width, height), color=(15, 23, 42))
        draw = ImageDraw.Draw(canvas)
        
        # Load fonts
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
            heading_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            title_font = heading_font = text_font = small_font = ImageFont.load_default()
        
        y_offset = 40
        
        # Title
        draw.text((width//2, y_offset), "Skin Lesion Analysis Report", 
                 fill=(125, 211, 252), font=title_font, anchor="mt")
        y_offset += 50
        draw.text((width//2, y_offset), f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                 fill=(148, 163, 184), font=small_font, anchor="mt")
        y_offset += 60
        
        # Prediction Box
        box_padding = 30
        box_height = 120
        draw.rectangle([(box_padding, y_offset), (width-box_padding, y_offset+box_height)], 
                      fill=(30, 41, 59), outline=(71, 85, 105), width=2)
        
        draw.text((box_padding+20, y_offset+20), f"Prediction: {prediction}", 
                 fill=(125, 211, 252), font=heading_font)
        draw.text((box_padding+20, y_offset+55), f"Confidence: {probability}%", 
                 fill=(52, 211, 153) if float(probability) >= 70 else (251, 191, 36), font=text_font)
        draw.text((box_padding+20, y_offset+85), f"Patient: Age {age}, {sex}, {site}", 
                 fill=(148, 163, 184), font=small_font)
        y_offset += box_height + 40
        
        # Disease Description
        if disease_info and prediction in disease_info:
            desc = disease_info[prediction]['description']
            desc_box_height = 100
            draw.rectangle([(box_padding, y_offset), (width-box_padding, y_offset+desc_box_height)], 
                          fill=(30, 41, 59), outline=(71, 85, 105), width=2)
            
            # Word wrap description
            words = desc.split()
            lines = []
            current_line = []
            for word in words:
                current_line.append(word)
                test_line = ' '.join(current_line)
                bbox = draw.textbbox((0, 0), test_line, font=small_font)
                if bbox[2] - bbox[0] > width - 2*box_padding - 40:
                    current_line.pop()
                    lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
            
            for i, line in enumerate(lines[:3]):  # Max 3 lines
                draw.text((box_padding+20, y_offset+15+i*25), line, 
                         fill=(203, 213, 225), font=small_font)
            y_offset += desc_box_height + 40
        
        # Images Section
        img_size = 320
        img_spacing = 30
        total_img_width = img_size * 4 + img_spacing * 3
        start_x = (width - total_img_width) // 2
        
        draw.text((width//2, y_offset), "Explainability Visualizations", 
                 fill=(125, 211, 252), font=heading_font, anchor="mt")
        y_offset += 50
        
        # Load and paste images
        images_data = [
            (original_path, "Original Image"),
            (lime_path, "LIME (End-to-End)"),
            (layercam_path, "Layer-CAM (Local)"),
            (attention_path, "Attention (Global)")
        ]
        
        for i, (img_path, label) in enumerate(images_data):
            x_pos = start_x + i * (img_size + img_spacing)
            
            if img_path and os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
                    
                    # Draw border
                    draw.rectangle([(x_pos-2, y_offset-2), (x_pos+img_size+2, y_offset+img_size+2)], 
                                  outline=(71, 85, 105), width=2)
                    
                    canvas.paste(img, (x_pos, y_offset))
                    
                    # Label
                    draw.text((x_pos + img_size//2, y_offset + img_size + 15), label, 
                             fill=(148, 163, 184), font=small_font, anchor="mt")
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
        
        y_offset += img_size + 60
        
        # Probability Chart (simplified bar chart)
        if all_probs:
            draw.text((width//2, y_offset), "All Class Probabilities", 
                     fill=(125, 211, 252), font=heading_font, anchor="mt")
            y_offset += 50
            
            labels = all_probs['labels']
            values = all_probs['values']
            
            chart_width = width - 2*box_padding - 100
            chart_height = 200
            chart_x = box_padding + 50
            max_bar_width = chart_width
            
            bar_height = chart_height // len(labels) - 5
            
            for i, (label, value) in enumerate(zip(labels, values)):
                bar_y = y_offset + i * (bar_height + 5)
                bar_width = int((value / 100) * max_bar_width)
                
                # Color based on value
                if value >= 70:
                    color = (52, 211, 153)
                elif value >= 50:
                    color = (251, 191, 36)
                elif value >= 30:
                    color = (96, 165, 250)
                else:
                    color = (148, 163, 184)
                
                # Draw bar
                draw.rectangle([(chart_x, bar_y), (chart_x + bar_width, bar_y + bar_height)], 
                              fill=color)
                
                # Label and value
                draw.text((chart_x - 40, bar_y + bar_height//2), label, 
                         fill=(203, 213, 225), font=small_font, anchor="rm")
                draw.text((chart_x + bar_width + 10, bar_y + bar_height//2), f"{value:.1f}%", 
                         fill=(203, 213, 225), font=small_font, anchor="lm")
            
            y_offset += chart_height + 60
        
        # Footer
        draw.text((width//2, height - 40), 
                 "This is for research purposes only. Consult medical professionals for diagnosis.", 
                 fill=(100, 116, 139), font=small_font, anchor="mt")
        
        return canvas
        
    except Exception as e:
        print(f"Error creating image report: {e}")
        traceback.print_exc()
        return None


# =========================================
# Routes
# =========================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Initial page (all None)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": None,
            "probability": None,
            "lime_img": None,
            "layercam_img": None,
            "attention_img": None,
            "original_img": None,
            "shap_contribs": None,
            "all_probs": None,
            "disease_info": DISEASE_INFO,
            "model_info": MODEL_INFO,
            "age": None,
            "sex": None,
            "site": None,
            "result_id": None,
        },
    )


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    site: str = Form(...)
):

    # Init placeholders
    pred_label = None
    pred_prob = None
    layercam_url = None
    attn_url = None
    shap_contribs = None

    # Cleanup old results (older than 1 hour)
    results_dir = os.path.join(BASE_DIR, "static", "results")
    cleanup_old_results(results_dir, max_age_seconds=3600)

    try:
        # Read image
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")

        # Image preprocess
        image_tensor = preprocess_image(image_pil).to(device)

        # Metadata preprocess
        meta_tensor = preprocess_metadata(
            age=age,
            sex=sex,
            site=site
        ).unsqueeze(0).to(device)

        # Attempt to capture attention maps from the ViT backbone using forward hooks.
        attn_maps = []
        hook_handles = []
        attn_qkv_outputs = []

        # Debug: report whether model has a ViT backbone
        try:
            has_vit = hasattr(model, 'vit') and getattr(model, 'vit') is not None
        except Exception:
            pass

        # If ViT exists, print a short list of modules that look relevant (qkv/attn/proj)
        try:
            if has_vit:
                candidates = []
                for name, m in getattr(model, 'vit').named_modules():
                    cls_name = m.__class__.__name__.lower()
                    lname = name.lower()
                    if ('qkv' in lname) or ('attn' in lname) or ('attention' in lname) or ('proj' in lname) or ('norm' in lname):
                        candidates.append((name, m.__class__.__name__))
                    # limit how many we print
                    if len(candidates) >= 40:
                        break
                pass
        except Exception as e:
            print('Error enumerating vit submodules:', e)

        def _extract_attn_tensors(x):
            """Recursively find tensors that look like attention maps: shape (B, heads, N, N) or (heads, N, N) or (B, N, N)"""
            found = []
            import torch as _torch
            if isinstance(x, _torch.Tensor):
                if x.dim() == 4 and x.shape[-2] == x.shape[-1]:
                    # (B, heads, N, N)
                    found.append(x)
                elif x.dim() == 3 and x.shape[-2] == x.shape[-1]:
                    # (heads, N, N) or (B, N, N) — convert to (1, heads, N, N) later
                    found.append(x)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    found.extend(_extract_attn_tensors(v))
            elif isinstance(x, dict):
                for v in x.values():
                    found.extend(_extract_attn_tensors(v))
            return found

        def _hook(module, inp, out):
            # inspect outputs and inputs for attention-like tensors
            for candidate in _extract_attn_tensors(out):
                try:
                    attn_maps.append(candidate.detach().cpu())
                except Exception:
                    pass
            for candidate in _extract_attn_tensors(inp):
                try:
                    # inp can be a tuple; ensure tensor
                    attn_maps.append(candidate.detach().cpu())
                except Exception:
                    pass

        def _qkv_hook(name):
            def fn(module, inp, out):
                try:
                    # out is qkv projection tensor: (B, N, 3*E)
                    attn_qkv_outputs.append((name, out.detach().cpu()))
                except Exception:
                    pass
            return fn

        # Register hooks on ViT attention-like submodules only (reduce false positives)
        try:
            for name, m in getattr(model, 'vit').named_modules():
                if m is getattr(model, 'vit'):
                    continue
                cls_name = m.__class__.__name__.lower()
                # common attention module identifiers in timm / pytorch implementations
                # Hook attention modules for possible direct attention outputs
                if ('attn' in cls_name) or ('attention' in cls_name) or ('multihead' in cls_name):
                    try:
                        h = m.register_forward_hook(_hook)
                        hook_handles.append(h)
                    except Exception:
                        pass
                # Additionally hook qkv linear layers specifically to reconstruct attention
                if 'qkv' in name or name.endswith('.qkv') or name.endswith('.attn.qkv'):
                    try:
                        h = m.register_forward_hook(_qkv_hook(name))
                        hook_handles.append(h)
                    except Exception:
                        pass
        except Exception:
            # If model has no vit attribute or modules can't be iterated, skip
            hook_handles = []

        with torch.no_grad():
            logits = None
            try:
                out = model(image_tensor, meta_tensor)
                # model may return (logits, attn1, attn2) or just logits
                if isinstance(out, (tuple, list)):
                    logits = out[0]
                    model_attn_outputs = list(out[1:])
                else:
                    logits = out
                    model_attn_outputs = []
            except Exception as e:
                # fallback to previous call signature
                logits, _, _ = model(image_tensor, meta_tensor)
                logits = logits
                model_attn_outputs = []

            probs = torch.softmax(logits, dim=1)

        # remove hooks
        for h in hook_handles:
            try:
                h.remove()
            except Exception:
                pass

        # Debug: print captured attention shapes (server log)
        # try:
        #     shapes = [tuple(a.shape) for a in attn_maps]
        #     if shapes:
        #         print('Captured attn maps (from hooks):', shapes)
        #     else:
        #         print('No attn maps captured from hooks')
        # except Exception:
        #     pass

        # If the model directly returned attention-like outputs, prefer those
        attn_url = None
        attn_candidates = []
        if 'model_attn_outputs' in locals() and model_attn_outputs:
            try:
                model_shapes = [tuple(a.shape) for a in model_attn_outputs if hasattr(a, 'shape')]
                # print('Model returned attention-like outputs shapes:', model_shapes)
            except Exception:
                pass
            for a in model_attn_outputs:
                try:
                    import torch as _torch
                    if isinstance(a, _torch.Tensor):
                        attn_candidates.append(a.detach().cpu())
                    elif isinstance(a, (list, tuple)):
                        for v in a:
                            if isinstance(v, _torch.Tensor):
                                attn_candidates.append(v.detach().cpu())
                except Exception:
                    pass

        # Fallback to captured hook attn_maps if model didn't return any
        if not attn_candidates and attn_maps:
            attn_candidates = attn_maps

        # If we captured qkv outputs from Linear layers, reconstruct attention matrices
        if attn_qkv_outputs:
            try:
                # print('Captured qkv outputs count:', len(attn_qkv_outputs))
                # build a mapping of module name -> module object for vit
                vit_modules = {n: m for n, m in getattr(model, 'vit').named_modules()} if hasattr(model, 'vit') else {}
                for name, qkv in attn_qkv_outputs:
                    try:
                        # qkv: (B, N, 3*E)
                        B, N, threeE = qkv.shape
                        if threeE % 3 != 0:
                            # print(f'Unexpected qkv last-dim not divisible by 3: {threeE} for {name}')
                            continue
                        E = threeE // 3
                        # try to find parent attn module to get num_heads
                        parent_name = name.rsplit('.', 1)[0]
                        parent = vit_modules.get(parent_name, None)
                        nheads = None
                        if parent is not None:
                            nheads = getattr(parent, 'num_heads', None) or getattr(parent, 'heads', None)
                        # fallback: try to infer heads by checking common values
                        if not nheads:
                            # try typical small heads for tiny vit
                            for h in (12, 8, 6, 4, 3, 2, 1):
                                if E % h == 0:
                                    nheads = h
                                    break
                        if not nheads:
                            # print(f'Unable to infer num_heads for qkv {name} (E={E}), skipping')
                            continue

                        head_dim = E // nheads
                        if head_dim * nheads != E:
                            # print(f'Inconsistent head_dim computation for {name}: E={E}, nheads={nheads}')
                            continue

                        # reshape and split
                        t = qkv.view(B, N, 3, nheads, head_dim)  # (B, N, 3, heads, head_dim)
                        t = t.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
                        q, k, v = t[0], t[1], t[2]  # each (B, heads, N, head_dim)

                        # compute attention: (B, heads, N, N)
                        import math
                        q = q.float()
                        k = k.float()
                        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
                        attn = torch.softmax(attn, dim=-1)

                        # append to candidates
                        attn_candidates.append(attn)
                    except Exception as e:
                        pass
                        continue
            except Exception as e:
                pass

        pred_idx = int(probs.argmax(dim=1).item())
        pred_prob = float(probs[0, pred_idx].item())
        pred_label = CLASS_NAMES[pred_idx]

        # Get all probabilities for chart
        all_probs_values = [float(probs[0, i].item() * 100) for i in range(len(CLASS_NAMES))]
        all_probs = {
            "labels": CLASS_NAMES,
            "values": all_probs_values
        }

        # Save resized original image (224x224 for display)
        results_dir = os.path.join(BASE_DIR, "static", "results")
        os.makedirs(results_dir, exist_ok=True)
        original_filename = f"original_{uuid.uuid4().hex}.png"
        original_path = os.path.join(results_dir, original_filename)
        image_pil.resize((224, 224)).save(original_path)

        # LIME Explanation (for Image Comparison)
        lime_url = generate_lime_explanation(
            model=model,
            img_tensor=image_tensor,
            meta_tensor=meta_tensor,
            original_image=image_pil,
            class_idx=pred_idx,
            device=device,
            base_dir=BASE_DIR,
            num_samples=500  # Adjust for speed vs accuracy
        )

        # Layer-CAM (CNN local features)
        layercam_url = generate_layercam(
            model=model,
            img_tensor=image_tensor,
            meta_tensor=meta_tensor,
            original_image=image_pil,
            class_idx=pred_idx,
            device=device,
            base_dir=BASE_DIR,
            layers=[model.convnext]  # Use ConvNeXt features (timm feature-extractor)
        )

        # Attention rollout (Transformer global features)
        # Use collected attn_candidates (preferred) or fallback to attn_maps
        candidates = attn_candidates if attn_candidates else attn_maps
        attn_url = None

        def _is_perfect_square(n: int) -> bool:
            if n <= 0: return False
            import math
            r = int(math.isqrt(n))
            return r * r == n

        if candidates:
            valid_layers = []
            
            for a in candidates:
                try:
                    import torch as _torch
                    if not isinstance(a, _torch.Tensor): continue

                    # 1. 統一維度到 (B, heads, N, N)
                    if a.dim() == 4:
                        t = a
                    elif a.dim() == 3:
                        heads_or_B = a.shape[0]
                        Ndim = a.shape[1]
                        # 判斷是 (Heads, N, N) 還是 (Batch, N, N)
                        if heads_or_B <= 32 and heads_or_B < Ndim: 
                            t = a.unsqueeze(0) # (1, heads, N, N)
                        else:
                            t = a.unsqueeze(1) # (B, 1, N, N)
                    elif a.dim() == 2:
                        t = a.unsqueeze(0).unsqueeze(0)
                    else:
                        continue

                    # 2. 檢查是否為方陣 (Attention Matrix 必須是方陣)
                    B, heads, N1, N2 = t.shape
                    if N1 != N2: continue
                    
                    # 3. 關鍵修改：不要切除 CLS Token，但要確認它的存在
                    # 我們只檢查形狀是否合理，保留完整矩陣給 Rollout 算
                    N = N1
                    is_pure_patches = _is_perfect_square(N)
                    has_cls_token = _is_perfect_square(N - 1)
                    
                    # 如果既不是完整方圖(如14x14=196)，也不是帶CLS的圖(197)，那就可能是 ConvNeXt 的特徵圖混進來了，跳過
                    if not (is_pure_patches or has_cls_token):
                        continue
                        
                    # 確保放到 CPU 以免累積 GPU 顯存
                    valid_layers.append(t.detach().cpu())

                except Exception:
                    continue

            # 4. 關鍵修改：不要只取前 6 層，Rollout 需要全層運算才準確
            # 如果真的記憶體不足要過濾，建議取 "最後" 幾層，而不是最前幾層
            # uniq = valid_layers[-6:] # 如果非要限制的話
            uniq = valid_layers 

            if uniq:
                try:
                    # 呼叫之前給你的、修正過 CLS 處理邏輯的 generate_attention_rollout
                    attn_url = generate_attention_rollout(
                        attn_layers=uniq,
                        original_image=image_pil,
                        base_dir=BASE_DIR
                    )
                except Exception as e:
                    print(f"Rollout generation failed: {e}")
                    attn_url = None

        # SHAP metadata
        shap_contribs = generate_shap_metadata(
            model=model,
            image_tensor=image_tensor,
            meta_tensor=meta_tensor,
            class_idx=pred_idx,
            device=device,
            top_k=None,          # show all non-zero
            filter_zero=True,
            zero_epsilon=1e-8,
        )

    except Exception as e:
        print("PREDICT ERROR:", e)
        traceback.print_exc()

    # Generate unique result ID for PDF download
    result_id = uuid.uuid4().hex[:8] if pred_label else None

    # Return to template
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": pred_label,
            "probability": f"{pred_prob*100:.2f}" if pred_prob is not None else None,
            "lime_img": lime_url if 'lime_url' in locals() else None,
            "layercam_img": layercam_url if 'layercam_url' in locals() else None,
            "attention_img": attn_url,
            "original_img": f"/static/results/{original_filename}" if 'original_filename' in locals() else None,
            "shap_contribs": shap_contribs,
            "all_probs": all_probs if 'all_probs' in locals() else None,
            "disease_info": DISEASE_INFO,
            "model_info": MODEL_INFO,
            "age": age,
            "sex": sex,
            "site": site,
            "result_id": result_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


@app.post("/download-report")
async def download_report(request: Request):
    """Generate and download a comprehensive image report"""
    try:
        # Get form data
        form_data = await request.form()
        
        results_dir = os.path.join(BASE_DIR, "static", "results")
        
        # Build full paths
        original_path = os.path.join(BASE_DIR, "static", form_data.get('original', '').lstrip("/static/"))
        lime_path = os.path.join(BASE_DIR, "static", form_data.get('lime', '').lstrip("/static/")) if form_data.get('lime') else None
        layercam_path = os.path.join(BASE_DIR, "static", form_data.get('layercam', '').lstrip("/static/")) if form_data.get('layercam') else None
        attention_path = os.path.join(BASE_DIR, "static", form_data.get('attention', '').lstrip("/static/")) if form_data.get('attention') else None
        
        # Parse probabilities and SHAP
        import json
        all_probs = json.loads(form_data.get('all_probs', '{}'))
        shap_contribs = json.loads(form_data.get('shap_contribs', '[]'))
        
        # Create image report
        report_img = create_image_report(
            original_path, lime_path, layercam_path, attention_path,
            form_data.get('prediction', ''),
            form_data.get('probability', ''),
            int(form_data.get('age', 0)),
            form_data.get('sex', ''),
            form_data.get('site', ''),
            all_probs,
            shap_contribs,
            DISEASE_INFO
        )
        
        if report_img:
            # Save image
            img_filename = f"report_{uuid.uuid4().hex[:8]}.png"
            img_path = os.path.join(results_dir, img_filename)
            
            report_img.save(img_path, "PNG", quality=95, optimize=True)
            
            return FileResponse(
                img_path,
                media_type="image/png",
                filename=f"skin_lesion_report_{form_data.get('prediction', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
        else:
            return {"error": "Failed to generate report"}
            
    except Exception as e:
        print(f"Download error: {e}")
        traceback.print_exc()
        return {"error": str(e)}
