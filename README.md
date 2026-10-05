# Mine Vehicle Safety AI
Academic AIML project for safe and efficient operation of mine vehicles in fog and low-visibility conditions.

## Project objective
This project is designed as a research prototype for driver-assistance in harsh road conditions. It combines:
- object detection for road users and obstacles,
- visibility classification under low-light and fog-like conditions,
- risk assessment for early warnings,
- a web dashboard for demonstration and academic reporting.

The system is not an autonomous vehicle controller and is meant for research and demonstration only.

## AIML workflow demonstrated in the project
This project is structured to reflect the major AIML stages:
1. Dataset acquisition and preparation
2. Data preprocessing and augmentation planning
3. Exploratory data analysis
4. Feature extraction
5. Model training
6. Validation and testing
7. Evaluation and reporting
8. Inference on live upload
9. Web deployment

The repository includes scripts and folder structure for the full pipeline, even when the large raw datasets are not committed to Git.

## Real-world dataset basis
This project uses the following public datasets as the primary basis for the training workflow:
- BDD100K for object detection and road/environment scenes
- RWVC-BDD100K for visibility labels

Official references:
- https://bdd-data.berkeley.edu/
- https://github.com/bdd100k/bdd100k
- https://github.com/enricivi/RWVC-BDD100K

Do not commit the full raw dataset into the repository. Instead, download it and place it in the required folders described below.

## Repository structure
```text
backend/
  app/
    main.py
    inference.py
    risk.py
    visibility.py
  training/
    build_subset.py
    prepare_detection_dataset.py
    train_detector.py
    evaluate_detector.py
    inference.py
    subset_config.json
    vehicle_detector.yaml
  models/
    vehicle_detector/
    visibility_rf.joblib

data/
  bdd100k/
  rwvc-bdd100k/
  processed/
  sample/
frontend/
  src/
```

## Dataset placement instructions
After downloading the datasets, place them in the following structure:

```text
data/
├── bdd100k/
│   ├── images/
│   │   └── 100k/
│   │       ├── train/
│   │       └── val/
│   └── labels/
│       ├── det_train.json
│       └── det_val.json
├── rwvc-bdd100k/
│   ├── images/
│   └── train_labels.json
├── processed/
│   ├── subset/
│   └── yolov8/
└── sample/
```

If you do not have the raw data yet, the project still runs in demo mode using the pretrained YOLOv8 fallback model and heuristic visibility scoring.

## Lightweight subset workflow
The project includes a configurable subset pipeline for a 2nd-year academic project. The main script is:

```bash
python backend/training/build_subset.py
```

This script reads the BDD100K annotations, keeps the configured subset size, avoids duplicates, preserves class coverage as much as practical, and writes:
- train/val/test image folders,
- label files,
- subset summary statistics.

The default subset size is defined in `backend/training/subset_config.json` and can be changed before running training:
- train: 3000
- val: 500
- test: 500

The summary is saved to:
`data/processed/subset/subset_summary.json`

## Object detection training workflow
The object detection pipeline is built around YOLOv8:
1. Create the subset with `build_subset.py`
2. Prepare the YOLO-formatted dataset with `prepare_detection_dataset.py`
3. Train the detector with `train_detector.py`
4. Evaluate it with `evaluate_detector.py`

Example:
```bash
python backend/training/build_subset.py
python backend/training/prepare_detection_dataset.py
python backend/training/train_detector.py
python backend/training/evaluate_detector.py
```

The training script supports:
- pretrained YOLOv8 starting weights,
- custom dataset,
- configurable epochs,
- configurable image size,
- configurable batch size,
- CPU/GPU selection,
- checkpoint saving,
- best-model saving.

The application automatically prefers the trained model at:
`backend/models/vehicle_detector/weights/best.pt`

If that model is missing, it falls back to the pretrained `yolov8n.pt` for demo/testing. This is clearly marked in the UI and backend status response.

## Visibility classification model
The visibility pipeline is built separately from the detector.

Recommended academic approach for this project:
- Random Forest classifier
- Image-derived features such as grayscale mean, grayscale standard deviation, contrast, Laplacian variance, edge density, brightness, and dark-pixel ratio
- RWVC-BDD100K visibility labels as the training target

The current implementation supports a trained Random Forest visibility model at:
`backend/models/visibility_rf.joblib`

When the model is missing, the project uses the heuristic OpenCV fallback and clearly reports that mode.

Policy used for final interpretation:
- GOOD = high visibility / low obscuration
- REDUCED = moderate degradation
- POOR = strong visibility degradation

The project maps binary `good/poor` training labels into the required academic 3-level safety policy by using a score threshold and confidence-based decision logic.

## Risk assessment logic
The risk engine combines:
- detected objects,
- object confidence,
- relative location in the frame,
- frame coverage / approximation of closeness,
- visibility level.

Output states:
- SAFE
- CAUTION
- HIGH RISK

## Run backend (PowerShell)
```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The first YOLO inference can automatically download `yolov8n.pt` if needed.

## Run frontend
```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL, normally:
`http://localhost:5173`

## Academic note
This project is a driver-assistance research prototype and is intentionally designed to be a realistic AIML workflow rather than a production safety controller.

It is suitable for a 2nd-year AIML academic project, with clear separation between:
- prototype demo mode,
- trained research pipeline,
- dataset-driven experimentation,
- explainable risk analysis.
