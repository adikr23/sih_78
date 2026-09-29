from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class BoundingBox4D(BaseModel):
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    t_start: int
    t_end: int
    n_points: int


class AlertOut(BaseModel):
    id: int
    event_name: str
    centroid_row: int
    centroid_col: int
    peak_value: float
    severity: str
    confidence: float
    radius_km: float
    physics_score: float
    efi_peak: float
    unet_peak: float
    diffusion_peak: float
    amplitude_gain_pct: float
    created_at: Optional[str] = None

    class Config:
        from_attributes = True


class PipelineRunRequest(BaseModel):
    event_name: str = "live_run"
    seed: Optional[int] = None


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | running | done | failed
    progress: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
