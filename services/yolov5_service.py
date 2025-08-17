# services/yolov5_service.py
import os
import cv2
import torch
from ultralytics import YOLO
import logging
from typing import Dict, List, Any
import numpy as np

logger = logging.getLogger(__name__)

class PotholeDetector:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load YOLO model for pothole detection"""
        try:
            # Option 1: Use pre-trained YOLOv8 model (you can fine-tune for potholes)
            self.model = YOLO('best.pt')  # or 'yolov8s.pt' for better accuracy
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            # Fallback: create a dummy model for testing
            self.model = None
    
    def detect_objects(self, image_path: str) -> List[Dict]:
        """Detect objects in image and filter for pothole-like objects"""
        if not self.model:
            # Return dummy detection for testing
            return self._dummy_detection()
        
        try:
            # Run inference
            results = self.model(image_path)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class name and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.model.names[class_id]
                        
                        # Filter for objects that could be potholes
                        # You'll need to train/fine-tune for actual pothole detection
                        pothole_like_classes = ['pothole', 'hole', 'crack', 'damage']
                        if any(keyword in class_name.lower() for keyword in pothole_like_classes) or confidence > 0.5:
                            detections.append({
                                'class': class_name,
                                'confidence': confidence,
                                'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                                'severity': self._calculate_severity(confidence, box.xyxy[0].tolist())
                            })
            
            return detections
        
        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return self._dummy_detection()
    
    def detect_in_video(self, video_path: str) -> List[Dict]:
        """Detect objects in video frames"""
        if not self.model:
            return self._dummy_detection()
        
        try:
            cap = cv2.VideoCapture(video_path)
            all_detections = []
            frame_count = 0
            
            # Process every 30th frame to avoid too many detections
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % 30 == 0:  # Process every 30th frame
                    # Save frame temporarily
                    temp_frame_path = f"temp_frame_{frame_count}.jpg"
                    cv2.imwrite(temp_frame_path, frame)
                    
                    # Detect in frame
                    frame_detections = self.detect_objects(temp_frame_path)
                    
                    # Add frame info to detections
                    for detection in frame_detections:
                        detection['frame'] = frame_count
                        detection['timestamp'] = frame_count / 30.0  # Assume 30 FPS
                    
                    all_detections.extend(frame_detections)
                    
                    # Clean up temp file
                    if os.path.exists(temp_frame_path):
                        os.remove(temp_frame_path)
                
                frame_count += 1
            
            cap.release()
            return all_detections
        
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return self._dummy_detection()
    
    def _calculate_severity(self, confidence: float, bbox: List[float]) -> str:
        """Calculate pothole severity based on confidence and size"""
        # Calculate bbox area (rough estimate of pothole size)
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        
        # Simple severity calculation (you can improve this)
        if confidence > 0.8 and area > 10000:
            return "Severe"
        elif confidence > 0.6 and area > 5000:
            return "Moderate"
        else:
            return "Minor"
    
    def _dummy_detection(self) -> List[Dict]:
        """Return dummy detection for testing when model is not available"""
        return [{
            'class': 'pothole',
            'confidence': 0.75,
            'bbox': [100, 100, 200, 200],
            'severity': 'Moderate'
        }]

# Global detector instance
detector = PotholeDetector()

def detect_pothole(image_path: str) -> bool:
    """Simple function to check if pothole exists in image"""
    detections = detector.detect_objects(image_path)
    return len(detections) > 0

def analyze_media_file(file_path: str, media_type: str) -> Dict[str, Any]:
    """Analyze media file for potholes and return detailed results"""
    try:
        if media_type == "image":
            detections = detector.detect_objects(file_path)
        elif media_type == "video":
            detections = detector.detect_in_video(file_path)
        else:
            raise ValueError(f"Unsupported media type: {media_type}")
        
        # Process results
        potholes_detected = len(detections) > 0
        detection_count = len(detections)
        severity_levels = [d['severity'] for d in detections]
        confidence_scores = [d['confidence'] for d in detections]
        
        return {
            "potholes_detected": potholes_detected,
            "detection_count": detection_count,
            "detections": detections,
            "severity_levels": severity_levels,
            "confidence_scores": confidence_scores,
            "media_type": media_type,
            "file_path": file_path
        }
    
    except Exception as e:
        logger.error(f"Error analyzing media file {file_path}: {e}")
        # Return safe default
        return {
            "potholes_detected": False,
            "detection_count": 0,
            "detections": [],
            "severity_levels": [],
            "confidence_scores": [],
            "media_type": media_type,
            "file_path": file_path,
            "error": str(e)
        }