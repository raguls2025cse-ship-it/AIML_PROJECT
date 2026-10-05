import json
from pathlib import Path

import cv2
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "visibility_rf.joblib"


def _extract_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    edges = cv2.Canny(gray, 80, 160)
    hist = np.histogram(gray, bins=16, range=(0, 256))[0]
    hist = hist / max(1.0, hist.sum())
    return np.array([
        float(np.mean(gray)),
        float(np.std(gray)),
        float(np.std(gray) ** 2),
        float(lap.var()),
        float(edges.mean()),
        float(np.mean(hist)),
        float(np.sum(gray < 30) / gray.size),
        float(np.mean(gray > 200)),
    ], dtype=np.float32)


def _heuristic_visibility_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast_n = min(1.0, contrast / 65.0)
    sharpness_n = min(1.0, sharpness / 500.0)
    brightness_n = 1.0 - min(1.0, abs(brightness - 125.0) / 125.0)
    score = 100.0 * (0.45 * contrast_n + 0.40 * sharpness_n + 0.15 * brightness_n)
    return round(max(0.0, min(100.0, score)), 2)


def _classify_with_model(frame):
    if not MODEL_PATH.exists():
        return None
    model = joblib.load(MODEL_PATH)
    features = _extract_features(frame)
    probability = model.predict_proba([features])[0]
    label = model.classes_[int(np.argmax(probability))]
    good_prob = float(probability[np.where(model.classes_ == "good")[0][0]]) if "good" in model.classes_ else 0.0
    poor_prob = float(probability[np.where(model.classes_ == "poor")[0][0]]) if "poor" in model.classes_ else 0.0
    score = max(0.0, min(100.0, 100.0 * (0.5 + (good_prob - poor_prob) * 0.5)))
    if good_prob >= 0.72:
        status = "GOOD"
    elif poor_prob >= 0.55:
        status = "POOR"
    else:
        status = "REDUCED"
    return score, status, "trained_visibility_model"


def visibility_score(frame):
    model_result = _classify_with_model(frame)
    if model_result is not None:
        score, status, _ = model_result
        return score, status
    return _heuristic_visibility_score(frame), visibility_status(_heuristic_visibility_score(frame))


def visibility_status(score):
    if score < 35: return "POOR"
    if score < 60: return "REDUCED"
    return "GOOD"
