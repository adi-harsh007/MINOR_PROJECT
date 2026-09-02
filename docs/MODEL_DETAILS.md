# Skin Cancer Classification Model Details

This document provides technical specifications and performance metrics for the skin cancer diagnostic model utilized in this project.

## Clinical Context: ABCDE Rules

Before specialized neural analysis, clinicians often use the **ABCDE criteria** for the visual assessment of pigmented lesions, particularly for identifying potential melanoma.

![Figure 1.1: ABCDE rules for visual clinical diagnosis of Melanoma](file:///d:/ML/MODEL_Skin-Cancer/docs/abcde_rules.png)
*Figure 1.1: Traditional ABCDE rules for visual clinical diagnosis of Melanoma.*

| Rule | Aspect | Description |
| :--- | :--- | :--- |
| **A** | Asymmetry | One half of the mole does not match the other. |
| **B** | Border | Edges are irregular, ragged, notched, or blurred. |
| **C** | Color | Pigmentation is not uniform across the lesion. |
| **D** | Diameter | Usually greater than 6mm (approx. size of a pencil eraser). |
| **E** | Evolving | The mole is changing in size, shape, or color. |

## Model Architecture

- **Base Model:** EfficientNet-B3 (via `timm` library)
- **Input Resolution:** 224x224 pixels
- **State:** Evaluation Mode (`eval()`) with pre-trained weights loaded from `models/latest.pt`.

## Data Processing & Preprocessing

The model uses a standardized ImageNet-based preprocessing pipeline:
1. **Resize:** Input images are resized to 256 pixels on the shorter side.
2. **Center Crop:** A 224x224 patch is cropped from the center.
3. **Normalization:**
   - Mean: `[0.485, 0.456, 0.406]`
   - Std Dev: `[0.229, 0.224, 0.225]`

## Classification Categories

The model is trained on the HAM10000 dataset to recognize the following 7 morphologies.

![Figure 1.4: 7-Class diagnostic samples from the training dataset](file:///d:/ML/MODEL_Skin-Cancer/docs/class_samples.png)
*Figure 1.4: Representative visual examples for each of the 7 clinical diagnostic classes.*

| Label | Full Name | Description |
| :--- | :--- | :--- |
| **akiec** | Actinic Keratosis | Also includes Bowen's disease / intraepithelial carcinoma. |
| **bcc** | Basal Cell Carcinoma | A common form of skin cancer. |
| **bkl** | Benign Keratosis | Includes seborrheic keratoses and lichen-planus like keratoses. |
| **df** | Dermatofibroma | A benign fibrous skin lesion. |
| **mel** | Melanoma | The most serious type of skin cancer. |
| **nv** | Melanocytic Nevi | Common moles (benign). |
| **vasc** | Vascular Lesions | Includes angiomas, hemorrhage, and pyogenic granulomas. |

## Performance Metrics

Metrics are based on optimized classification thresholds to maximize the Macro F1-Score.

### Baseline vs. Optimized Performance
- **Baseline F1 (Macro):** 0.7272
- **Optimized F1 (Macro):** 0.7288

### Confusion Matrix (7x7)

The following heatmap was programmatically generated from the model's performance metadata, illustrating the normalized relationship between True Positives and False Positives.

![Figure 1.3: Automated 7x7 Confusion Matrix](file:///d:/ML/MODEL_Skin-Cancer/docs/confusion_matrix_automated.png)
*Figure 1.3: Automated confusion matrix showing normalized diagnostic accuracy across 7 classes.*

### Training & Convergence Analytics

The following charts demonstrate the model's learning progress and stabilization across the training session.

![Figure 1.5: Automated Training vs Validation Loss](file:///d:/ML/MODEL_Skin-Cancer/docs/loss_curve_automated.png)
*Figure 1.5: Cross-entropy loss convergence showing successful training stabilization.*

![Figure 1.6: Automated Macro F1-Score Progression](file:///d:/ML/MODEL_Skin-Cancer/docs/f1_curve_automated.png)
*Figure 1.6: Macro F1-Score progression reaching the clinical target of ~0.73.*

![Figure 1.8: Per-Epoch Accuracy Progression](file:///d:/ML/MODEL_Skin-Cancer/docs/accuracy_curve_automated.png)
*Figure 1.8: Granular assessment of Top-1 Accuracy across every training epoch.*

### Training vs. Testing: Final Performance Summary

To ensure clinical reliability and lack of overfitting, the final model was evaluated on both the training set and a hold-out testing set.

![Figure 1.7: Training vs Testing Performance Comparison](file:///d:/ML/MODEL_Skin-Cancer/docs/performance_summary_automated.png)
*Figure 1.7: Comparison of Accuracy, Macro-F1, Precision, and Recall across Training and Testing sessions.*

| Class | Optimized Threshold | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **akiec** | 0.4000 | 0.6154 | 0.5000 | 0.5517 |
| **bcc** | 0.4500 | 0.6875 | 0.8148 | 0.7458 |
| **bkl** | 0.4000 | 0.7184 | 0.7440 | 0.7310 |
| **df** | 0.3000 | 0.8462 | 0.6111 | 0.7097 |
| **mel** | 0.2500 | 0.6316 | 0.5714 | 0.6000 |
| **nv** | 0.3500 | 0.9319 | 0.9444 | 0.9382 |
| **vasc** | 0.3000 | 0.8966 | 0.7647 | 0.8254 |

## Intelligent Out-of-Distribution (OOD) Gatekeeper

The system implements a two-stage gatekeeper to ensure only valid skin images are processed.

![Figure 1.2: OOD Gatekeeper & Decision Logic Flowchart](file:///d:/ML/MODEL_Skin-Cancer/docs/logic_flowchart.png)
*Figure 1.2: Logic flow for Out-of-Distribution detection and final acceptance.*

### 1. Color Profile Analysis
Before inference, the image is analyzed for "skin-like" characteristics:
- **Skin Hue Ratio:** Must be > 60% (detects non-skin colors).
- **Blue-Green Ratio:** Must be < 25% (detects medical backgrounds or clothes).
- **Saturation Check:** Rejects grayscale or near-grayscale images (sat < 15% should not exceed 70% of pixels).
- **Average Standard Deviation:** Rejects images that are too uniform (avg_std < 7.0) or too noisy (avg_std > 65.0).

### 2. Confidence Margin Analysis
Post-inference, the maximum margin over the class-specific threshold is checked:
- **Decision Rule:** `Prediction = argmax(Prob - Threshold)`
- **Low Confidence Rejection:** If the maximum margin is negative (`best_margin < 0`), the image is rejected as "Low Confidence" even if it passed the color gate.

---
*Last Updated: 2026-03-27*
