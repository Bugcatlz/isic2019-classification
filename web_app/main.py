import os
import uuid
import glob
from io import BytesIO
from typing import List
import io
import traceback
import time

import torch
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from models import build_model  # Hybrid model
import config
from .utils.preprocess import preprocess_image, preprocess_metadata
from .explainer.layercam import generate_layercam
from .explainer.attention import generate_attention_rollout
from .explainer.shap_meta import generate_shap_metadata


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
MODEL_PATH = os.path.join(PARENT_DIR, config.checkpoint_folder, "best_model.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"

# meta_dim = 1(age) + 1(sex) + 8(anatom_site)
model = build_model(num_classes=len(CLASS_NAMES), meta_dim=10).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


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
            "layercam_img": None,
            "attention_img": None,
            "original_img": None,
            "shap_contribs": None,
            "all_probs": None,
            "disease_info": DISEASE_INFO,
            "age": None,
            "sex": None,
            "site": None,
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

        # Forward pass
        with torch.no_grad():
            logits, attn1, attn2 = model(image_tensor, meta_tensor)
            probs = torch.softmax(logits, dim=1)

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

        # LayerCAM
        layercam_url = generate_layercam(
            model=model,
            img_tensor=image_tensor,
            meta_tensor=meta_tensor,
            original_image=image_pil,
            class_idx=pred_idx,
            device=device,
            base_dir=BASE_DIR,
            layers=[model.cnn1.conv, model.cnn2.conv]
        )

        # Attention rollout
        attn_url = generate_attention_rollout(
            attn_layers=[attn1, attn2],
            original_image=image_pil,
            base_dir=BASE_DIR
        )

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

    # Return to template
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": pred_label,
            "probability": f"{pred_prob*100:.2f}" if pred_prob is not None else None,
            "layercam_img": layercam_url,
            "attention_img": attn_url,
            "original_img": f"/static/results/{original_filename}" if 'original_filename' in locals() else None,
            "shap_contribs": shap_contribs,
            "all_probs": all_probs if 'all_probs' in locals() else None,
            "disease_info": DISEASE_INFO,
            "age": age,
            "sex": sex,
            "site": site,
        },
    )

