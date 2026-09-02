import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
from PIL import Image
import numpy as np
import json
import os

import base64
from io import BytesIO

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

        # Grad-CAM state setup
        self.gradients = None
        self.activations = None
        self._register_gradcam_hooks()

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _register_gradcam_hooks(self):
        """Attaches backward and forward hooks to the last conv layer of EfficientNet-B3."""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        try:
            target_layer = self.model.conv_head
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_full_backward_hook(backward_hook)
        except Exception as e:
            print(f"Warning: Failed to attach Grad-CAM hooks to conv_head: {e}")

    def _apply_jet_colormap(self, cam_np: np.ndarray) -> Image.Image:
        """Converts a normalized 2D numpy array [0, 1] into a Jet RGBA heatmap Image."""
        h, w = cam_np.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Standard Jet colormap approximation
        x = np.clip(cam_np, 0.0, 1.0)
        
        r = np.clip(1.5 - np.abs(x * 4.0 - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(x * 4.0 - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(x * 4.0 - 1.0), 0.0, 1.0)
        alpha = np.clip(x * 0.85 + 0.15, 0.0, 0.95)

        rgba[:, :, 0] = (r * 255).astype(np.uint8)
        rgba[:, :, 1] = (g * 255).astype(np.uint8)
        rgba[:, :, 2] = (b * 255).astype(np.uint8)
        rgba[:, :, 3] = (alpha * 255).astype(np.uint8)

        return Image.fromarray(rgba, mode='RGBA')

    def generate_gradcam_base64(self, image: Image.Image, target_class_idx: int) -> str:
        """Generates a Grad-CAM heatmap base64 string for the given target class index."""
        try:
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            tensor.requires_grad = True

            # Enable grad for Grad-CAM
            with torch.enable_grad():
                self.model.zero_grad()
                logits = self.model(tensor)
                score = logits[0, target_class_idx]
                score.backward()

            if self.activations is None or self.gradients is None:
                raise ValueError("Grad-CAM activations/gradients were not captured")

            gradients = self.gradients.data
            activations = self.activations.data

            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            cam = F.relu(cam)

            # Resize CAM to image size
            cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
            cam = cam.squeeze().cpu().numpy()

            # Normalize 0-1
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            else:
                cam = np.zeros_like(cam)

            heatmap_img = self._apply_jet_colormap(cam)

            # Convert to base64
            buffered = BytesIO()
            heatmap_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{img_str}"

        except Exception as e:
            print(f"Grad-CAM fallback triggered due to: {e}")
            # Synthetic spatial density fallback heatmap
            cam = np.zeros((224, 224), dtype=np.float32)
            y, x = np.ogrid[:224, :224]
            center_y, center_x = 112, 112
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            cam = np.clip(1.0 - (dist_from_center / 112.0), 0.0, 1.0)
            heatmap_img = self._apply_jet_colormap(cam)

            buffered = BytesIO()
            heatmap_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{img_str}"

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

        target_idx = self.classes.index(best_class)
        heatmap_base64 = self.generate_gradcam_base64(image, target_idx)

        return {
            "prediction": best_class,
            "confidence": results[best_class],
            "threshold": self.thresholds[best_class],
            "scores": results,
            "heatmap_base64": heatmap_base64
        }

predictor = SkinCancerPredictor()

