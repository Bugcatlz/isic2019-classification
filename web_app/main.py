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
MODEL_PATH = os.path.join(PARENT_DIR, config.checkpoint_folder, "best_model.pth")
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
            layers=[model.cnn2]  # Use CNN features
        )

        # Attention rollout (Transformer global features)
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
