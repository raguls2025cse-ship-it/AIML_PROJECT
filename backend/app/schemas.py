from pydantic import BaseModel
from typing import List, Optional

class Detection(BaseModel):
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    risk: str

class AnalysisResponse(BaseModel):
    filename: str
    media_type: str
    visibility_score: float
    visibility_status: str
    risk_level: str
    warning: str
    object_count: int
    detections: List[Detection]
    annotated_file: str
    frames_analyzed: Optional[int] = None
