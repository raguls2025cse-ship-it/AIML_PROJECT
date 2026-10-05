import os
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .inference import detect_frame

BASE = os.path.dirname(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app = FastAPI(title="Mine Vehicle Safety AI", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/files", StaticFiles(directory=UPLOAD_DIR), name="files")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "mine-vehicle-safety-ai",
        "model_mode": "trained model if available else fallback pretrained YOLOv8 demo",
        "dataset_basis": "BDD100K + RWVC-BDD100K",
    }


@app.get("/api/model-status")
def model_status():
    trained_model = os.path.exists(os.path.join(BASE, "models", "vehicle_detector", "weights", "best.pt"))
    visibility_model = os.path.exists(os.path.join(BASE, "models", "visibility_rf.joblib"))
    return {
        "detector_model": "trained" if trained_model else "pretrained-fallback",
        "visibility_model": "trained" if visibility_model else "heuristic-fallback",
        "trained_model_path": os.path.join(BASE, "models", "vehicle_detector", "weights", "best.pt") if trained_model else None,
        "visibility_model_path": os.path.join(BASE, "models", "visibility_rf.joblib") if visibility_model else None,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")
    ext = os.path.splitext(file.filename or "")[1].lower()
    job = uuid.uuid4().hex

    if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        frame = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(400, "Could not decode image")
        annotated, score, status, risk, warning, detections = detect_frame(frame)
        name = f"{job}.jpg"
        cv2.imwrite(os.path.join(UPLOAD_DIR, name), annotated)
        return {
            "filename": file.filename,
            "media_type": "image",
            "visibility_score": round(float(score), 2),
            "visibility_status": status,
            "risk_level": risk,
            "warning": warning,
            "object_count": len(detections),
            "detections": detections,
            "annotated_file": f"/files/{name}",
            "frames_analyzed": 1,
        }

    if ext not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(400, "Supported: JPG PNG WEBP MP4 AVI MOV MKV")

    inp = os.path.join(UPLOAD_DIR, f"{job}_input{ext}")
    with open(inp, "wb") as fh:
        fh.write(content)

    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        raise HTTPException(400, "Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    name = f"{job}.mp4"
    out = os.path.join(UPLOAD_DIR, name)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    every = max(1, int(fps // 2))
    n = 0
    vals = []
    all_d = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % every == 0:
            annotated, score, status, risk, warning, dets = detect_frame(frame)
            vals.append(float(score))
            all_d.extend(dets)
            writer.write(annotated)
        else:
            writer.write(frame)
        n += 1

    cap.release()
    writer.release()

    if not vals:
        raise HTTPException(400, "No frames found")

    score = round(sum(vals) / len(vals), 2)
    avg_status = "POOR" if score < 35 else "REDUCED" if score < 60 else "GOOD"
    risk = "HIGH RISK" if any(d["risk"] == "HIGH" for d in all_d) else ("CAUTION" if avg_status != "GOOD" or any(d["risk"] == "CAUTION" for d in all_d) else "SAFE")
    warning_map = {
        "HIGH RISK": "OBJECT DETECTED IN VEHICLE DANGER ZONE",
        "CAUTION": "LOW VISIBILITY OR POTENTIAL HAZARD DETECTED",
        "SAFE": "NO IMMEDIATE HIGH-RISK CONDITION DETECTED",
    }
    warning = warning_map.get(risk, "LOW RISK")
    unique = {}
    for d in all_d:
        key = (d["label"], d["risk"])
        if key not in unique or d["confidence"] > unique[key]["confidence"]:
            unique[key] = d

    return {
        "filename": file.filename,
        "media_type": "video",
        "visibility_score": score,
        "visibility_status": avg_status,
        "risk_level": risk,
        "warning": warning,
        "object_count": len(all_d),
        "detections": list(unique.values()),
        "annotated_file": f"/files/{name}",
        "frames_analyzed": len(vals),
    }
