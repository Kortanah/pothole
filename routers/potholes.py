# routers/potholes.py
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from database import get_db
from models import Pothole, User
from schemas import PotholeOut, PotholeAnalysisOut, PotholeCaptureOut
from services.yolov5_service import detect_pothole, analyze_media_file
from utils.security import get_current_user
import shutil
import os
import uuid
from typing import Optional, List
import logging

router = APIRouter(prefix="/potholes", tags=["Potholes"])

UPLOAD_DIR = "uploads"
ANALYSIS_DIR = "analysis"  # For uploaded files (no location)
CAPTURE_DIR = "captures"   # For live captures (with location)
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB for videos

# Create directories
for directory in [UPLOAD_DIR, ANALYSIS_DIR, CAPTURE_DIR]:
    os.makedirs(directory, exist_ok=True)

logger = logging.getLogger(__name__)

def validate_media_file(file: UploadFile, allow_video: bool = False) -> str:
    """Validate uploaded media file and return file type"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_ext = os.path.splitext(file.filename.lower())[1]
    
    if file_ext in ALLOWED_IMAGE_EXTENSIONS:
        file_type = "image"
    elif allow_video and file_ext in ALLOWED_VIDEO_EXTENSIONS:
        file_type = "video"
    else:
        allowed = list(ALLOWED_IMAGE_EXTENSIONS)
        if allow_video:
            allowed.extend(ALLOWED_VIDEO_EXTENSIONS)
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(allowed)}"
        )
    
    # Check file size
    file.file.seek(0, os.SEEK_END)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    return file_type

# ============ MODE 1: ANALYSIS MODE (Upload existing files, no location) ============

@router.post("/analyze", response_model=PotholeAnalysisOut)
async def analyze_media(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    ANALYSIS MODE: Analyze uploaded image/video for potholes
    - Accepts existing media files
    - AI detects potholes
    - NO location data captured
    - Results stored for reference only
    """
    
    # Validate file (allow both images and videos)
    file_type = validate_media_file(file, allow_video=True)
    
    # Generate unique filename
    file_ext = os.path.splitext(file.filename)[1]
    unique_filename = f"analysis_{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(ANALYSIS_DIR, unique_filename)
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Analyze media for potholes
        analysis_result = analyze_media_file(file_path, file_type)
        
        logger.info(f"Analysis completed by user {current_user.id}: {analysis_result}")
        
        return {
            "file_path": file_path,
            "file_type": file_type,
            "analysis_result": analysis_result,
            "potholes_detected": analysis_result["potholes_detected"],
            "detection_count": analysis_result["detection_count"],
            "severity_levels": analysis_result["severity_levels"],
            "confidence_scores": analysis_result["confidence_scores"],
            "has_location": False,
            "mode": "analysis"
        }
        
    except Exception as e:
        # Clean up file on error
        if os.path.exists(file_path):
            os.remove(file_path)
        
        logger.error(f"Error during media analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Error analyzing media file")

# ============ MODE 2: LIVE CAPTURE MODE (Real-time capture with GPS location) ============

@router.post("/capture", response_model=PotholeCaptureOut)
async def capture_and_detect(
    file: UploadFile = File(...),
    latitude: float = Form(...),  # Required for live capture
    longitude: float = Form(...), # Required for live capture
    accuracy: Optional[float] = Form(None),  # GPS accuracy in meters
    timestamp: Optional[str] = Form(None),   # When the photo/video was taken
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    LIVE CAPTURE MODE: Capture photo/video with real-time location
    - Takes photo/video using device camera
    - Automatically captures GPS coordinates
    - AI detects potholes with location context
    - Creates pothole reports for authorities
    """
    
    # Validate file (allow both images and videos)
    file_type = validate_media_file(file, allow_video=True)
    
    # Validate coordinates
    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        raise HTTPException(
            status_code=400, 
            detail="Invalid GPS coordinates"
        )
    
    # Generate unique filename for live capture
    file_ext = os.path.splitext(file.filename)[1]
    unique_filename = f"capture_{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(CAPTURE_DIR, unique_filename)
    
    try:
        # Save captured file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Detect potholes in captured media
        detection_result = analyze_media_file(file_path, file_type)
        
        # Only create database records if potholes are detected
        pothole_records = []
        if detection_result["potholes_detected"]:
            
            # Create pothole record for each detected pothole
            for i, detection in enumerate(detection_result["detections"]):
                new_pothole = Pothole(
                    user_id=current_user.id,
                    image_path=file_path,
                    media_type=file_type,
                    latitude=latitude,
                    longitude=longitude,
                    gps_accuracy=accuracy,
                    capture_timestamp=timestamp,
                    status="Pending",
                    severity=detection["severity"],
                    confidence=detection["confidence"],
                    detection_index=i  # For multiple detections in same media
                )
                
                db.add(new_pothole)
                pothole_records.append(new_pothole)
            
            db.commit()
            
            # Refresh all records
            for pothole in pothole_records:
                db.refresh(pothole)
        
        logger.info(f"Live capture by user {current_user.id}: {len(pothole_records)} potholes detected at ({latitude}, {longitude})")
        
        return {
            "file_path": file_path,
            "file_type": file_type,
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "accuracy": accuracy
            },
            "analysis_result": detection_result,
            "potholes_detected": detection_result["potholes_detected"],
            "detection_count": detection_result["detection_count"],
            "severity_levels": detection_result["severity_levels"],
            "confidence_scores": detection_result["confidence_scores"],
            "pothole_records": [{"id": p.id, "severity": p.severity} for p in pothole_records],
            "has_location": True,
            "mode": "live_capture"
        }
        
    except Exception as e:
        # Clean up file on error
        if os.path.exists(file_path):
            os.remove(file_path)
        
        if isinstance(e, HTTPException):
            raise e
        
        logger.error(f"Error during live capture: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing live capture")

# ============ EXISTING ENDPOINTS (Modified) ============

@router.get("/reports", response_model=List[PotholeOut])
def get_pothole_reports(
    skip: int = 0, 
    limit: int = 100,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get pothole reports (only from live captures with location)"""
    query = db.query(Pothole)
    
    if status:
        query = query.filter(Pothole.status == status)
    
    if severity:
        query = query.filter(Pothole.severity == severity)
    
    return query.offset(skip).limit(limit).all()

@router.get("/reports/{pothole_id}", response_model=PotholeOut)
def get_pothole_report(pothole_id: int, db: Session = Depends(get_db)):
    """Get specific pothole report by ID"""
    pothole = db.query(Pothole).filter(Pothole.id == pothole_id).first()
    if not pothole:
        raise HTTPException(status_code=404, detail="Pothole report not found")
    return pothole

@router.put("/reports/{pothole_id}/status")
def update_pothole_status(
    pothole_id: int,
    status: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update pothole report status (for authorities)"""
    pothole = db.query(Pothole).filter(Pothole.id == pothole_id).first()
    if not pothole:
        raise HTTPException(status_code=404, detail="Pothole report not found")
    
    valid_statuses = ["Pending", "In Progress", "Fixed", "Rejected"]
    if status not in valid_statuses:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
        )
    
    pothole.status = status
    db.commit()
    db.refresh(pothole)
    
    return {"message": f"Pothole report {pothole_id} status updated to {status}"}

@router.get("/my-reports", response_model=List[PotholeOut])
def get_my_pothole_reports(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user's pothole reports"""
    return db.query(Pothole).filter(Pothole.user_id == current_user.id).all()