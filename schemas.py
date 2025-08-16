# schemas.py
# Defines request/response Pydantic models for validation
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

# -------------------- USER --------------------
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    password: str = Field(..., min_length=8)

class UserOut(BaseModel):
    id: int
    username: str
    email: str
    is_admin: bool
    
    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# -------------------- POTHOLE --------------------
class PotholeBase(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class PotholeCreate(PotholeBase):
    pass

class PotholeOut(PotholeBase):
    id: int
    user_id: int
    image_path: str
    media_type: Optional[str] = "image"  # "image" or "video"
    status: str
    severity: Optional[str] = None
    confidence: Optional[float] = None
    gps_accuracy: Optional[float] = None
    capture_timestamp: Optional[str] = None
    detection_index: Optional[int] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

# -------------------- ANALYSIS RESPONSES --------------------
class PotholeAnalysisOut(BaseModel):
    """Response for analysis mode (no location)"""
    file_path: str
    file_type: str  # "image" or "video"
    analysis_result: dict
    potholes_detected: bool
    detection_count: int
    severity_levels: list[str]
    confidence_scores: list[float]
    has_location: bool = False
    mode: str = "analysis"

class LocationInfo(BaseModel):
    latitude: float
    longitude: float
    accuracy: Optional[float] = None

class PotholeCaptureOut(BaseModel):
    """Response for live capture mode (with location)"""
    file_path: str
    file_type: str  # "image" or "video"
    location: LocationInfo
    analysis_result: dict
    potholes_detected: bool
    detection_count: int
    severity_levels: list[str]
    confidence_scores: list[float]
    pothole_records: list[dict]  # Created database records
    has_location: bool = True
    mode: str = "live_capture"

# -------------------- RESPONSES --------------------
class StatusResponse(BaseModel):
    message: str
    success: bool = True