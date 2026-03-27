# DermaScan AI — Clinical Diagnostic Engine

DermaScan AI is a clinical-grade skin lesion classification system based on an EfficientNet-B3 architecture trained on the ISIC/HAM10000 dataset.

## Project Structure

```text
/
├── backend/            # FastAPI source code and logic
├── frontend/           # Web interface (HTML/JS)
├── models/             # Model weights and optimal thresholds
├── data/               # Clinical database and lesion uploads
├── tests/              # OOD and functional verification scripts
├── docs/               # Technical documentation and history
├── samples/            # Representative test images
└── start.py            # Unified startup script
```

## Setup & Running

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start the Application**:
    ```bash
    python start.py
    ```
    Access the terminal at `http://localhost:8088`.

## Key Features

- **EfficientNet-B3 Backbone**: High-accuracy multiclass classification (7 classes).
- **Out-of-Distribution (OOD) Gatekeeper**: Dual-layer rejection system (HSV color profiling + confidence thresholding) to prevent misdiagnosis of non-skin images.
- **Persistent Clinical History**: Lightweight SQLite storage for tracking previous scans.
- **Authentication-Free**: Optimized for local clinical clinics and direct diagnostic workflows.
