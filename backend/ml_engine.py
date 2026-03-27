import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
from PIL import Image
import numpy as np
import json
import os

from .config import MODEL_PATH, THRESHOLD_PATH

class SkinCancerPredictor:
    def __init__(self, model_path=MODEL_PATH, threshold_path=THRESHOLD_PATH):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        
        # Load thresholds
        try:
            with open(threshold_path, 'r') as f:
                data = json.load(f)
                self.thresholds = {k: data['per_class_metrics'][k]['threshold'] for k in self.classes}
        except Exception as e:
            print(f"Warning: Could not load exact thresholds. Using defaults. ({e})")
            self.thresholds = {k: 0.5 for k in self.classes}
            
        # Initialize EfficientNet-B3
        self.model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=len(self.classes))
        
        try:
            ckpt = torch.load(model_path, map_location=self.device)
            state_dict = ckpt.get('model_state_dict', ckpt)
            cleaned_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(cleaned_dict)
            print(f"Model successfully loaded on {self.device}")
        except Exception as e:
            print(f"Error loading PyTorch weights: {e}")
            
        self.model.to(self.device).eval()
        print("OOD Gatekeeper: Color Profile Analysis active")

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _is_skin_image(self, pil_image: Image.Image) -> dict:
        """
        Returns a dictionary with OOD status and metrics.
        """
        img = pil_image.resize((128, 128))
        hsv = np.array(img.convert('HSV'))
        
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        total_pixels = h.size
        
        skin_hue_mask = (h < 50) | (h > 200)
        skin_hue_ratio = float(np.sum(skin_hue_mask) / total_pixels)
        
        strong_blue_green = (h > 60) & (h < 170) & (s > 60)
        blue_green_ratio = float(np.sum(strong_blue_green) / total_pixels)
        
        low_sat_ratio = float(np.sum(s < 15) / total_pixels)
        
        rgb = np.array(pil_image.resize((128, 128))).astype(np.float32)
        channel_stds = rgb.std(axis=(0, 1))
        avg_std = float(channel_stds.mean())
        
        metrics = {
            "skin_hue": skin_hue_ratio,
            "blue_green": blue_green_ratio,
            "low_sat": low_sat_ratio,
            "avg_std": avg_std
        }

        if skin_hue_ratio < 0.60:
            return {"is_ood": True, "reason": "insufficient_skin_hue"}
        if blue_green_ratio > 0.25:
            return {"is_ood": True, "reason": "excessive_non_skin_colors"}
        if low_sat_ratio > 0.70:
            return {"is_ood": True, "reason": "grayscale_not_allowed"}
        if avg_std < 7.0:
            return {"is_ood": True, "reason": "too_uniform"}
        if avg_std > 65.0:
            return {"is_ood": True, "reason": "high_frequency_noise"}
            
        return {"is_ood": False}

    def predict(self, image: Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # ── OOD Gate: Color Profile Analysis ─────────────────────
        ood_check = self._is_skin_image(image)
        if ood_check["is_ood"]:
            return ood_check
            
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy().tolist()
            
        results = {c: probs[i] for i, c in enumerate(self.classes)}
        
        # Classify by maximum margin over the optimal threshold
        best_class = None
        best_margin = -999.0
        
        for c, p in results.items():
            margin = p - self.thresholds[c]
            if margin > best_margin:
                best_margin = margin
                best_class = c

        # ── OOD Check: Score Rejection ────────────────────────────
        if best_margin < 0:
            return {"is_ood": True, "reason": "low_confidence"}

        return {
            "prediction": best_class,
            "confidence": results[best_class],
            "threshold": self.thresholds[best_class],
            "scores": results
        }

predictor = SkinCancerPredictor()
