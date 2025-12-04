import os
import uuid
import numpy as np
import torch
from PIL import Image
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt


def generate_lime_explanation(model, img_tensor, meta_tensor, original_image, 
                              class_idx, device, base_dir, num_samples=1000):
    """
    Generate LIME explanation for the prediction
    
    Args:
        model: The trained model
        img_tensor: Preprocessed image tensor [1, 3, H, W]
        meta_tensor: Metadata tensor [1, meta_dim]
        original_image: PIL Image (original)
        class_idx: Predicted class index
        device: torch device
        base_dir: Base directory for saving
        num_samples: Number of samples for LIME
    
    Returns:
        URL path to the saved LIME visualization
    """
    try:
        # Prepare the image for LIME (numpy array, RGB, 0-255)
        img_np = np.array(original_image.resize((224, 224)))
        
        # Define prediction function for LIME
        def predict_fn(images):
            """
            Prediction function for LIME
            images: numpy array of shape (batch, H, W, 3) with values 0-255
            """
            batch_size = images.shape[0]
            predictions = []
            
            model.eval()
            with torch.no_grad():
                for img in images:
                    # Convert to PIL and preprocess
                    img_pil = Image.fromarray(img.astype('uint8'))
                    
                    # Use the same preprocessing as in main
                    from ..utils.preprocess import preprocess_image
                    img_preprocessed = preprocess_image(img_pil).to(device)
                    
                    # Repeat metadata for this sample
                    meta_batch = meta_tensor.repeat(1, 1)
                    
                    # Forward pass
                    logits, _, _ = model(img_preprocessed, meta_batch)
                    probs = torch.softmax(logits, dim=1)
                    predictions.append(probs.cpu().numpy()[0])
            
            return np.array(predictions)
        
        # Create LIME explainer
        explainer = lime_image.LimeImageExplainer()
        
        # Generate explanation
        explanation = explainer.explain_instance(
            img_np,
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples,
            batch_size=10
        )
        
        # Get the explanation for the predicted class
        temp, mask = explanation.get_image_and_mask(
            class_idx,
            positive_only=True,
            num_features=10,
            hide_rest=False
        )
        
        # Create heatmap from mask
        heatmap = mask.astype(np.float32)
        
        # Normalize to 0-255
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            heatmap = heatmap.astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        overlay = cv2.addWeighted(img_np, 0.5, heatmap_colored, 0.5, 0)
        
        # Convert to PIL Image
        overlay_pil = Image.fromarray(overlay)
        
        # Save
        results_dir = os.path.join(base_dir, "static", "results")
        os.makedirs(results_dir, exist_ok=True)
        
        filename = f"lime_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(results_dir, filename)
        
        overlay_pil.save(filepath)
        
        return f"/static/results/{filename}"
        
    except Exception as e:
        print(f"LIME generation error: {e}")
        import traceback
        traceback.print_exc()
        return None
