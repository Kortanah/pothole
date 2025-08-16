# services/yolov5_service.py
import logging
import os
import random
from typing import Dict, List

logger = logging.getLogger(__name__)

def detect_pothole(image_path: str) -> str:
    """
    Legacy function for backward compatibility
    Detect potholes in a single image and return severity level
    """
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return "None"
        
        logger.info(f"Processing single image: {image_path}")
        
        # Mock detection based on filename
        filename = os.path.basename(image_path).lower()
        
        if "severe" in filename or "bad" in filename:
            return "Severe"
        elif "moderate" in filename or "medium" in filename:
            return "Moderate"
        elif "none" in filename or "good" in filename:
            return "None"
        else:
            # Random detection with realistic distribution
            result = random.choice(["Moderate", "Severe", "Moderate", "None"])
            logger.info(f"Mock detection result: {result}")
            return result
            
    except Exception as e:
        logger.error(f"Error during pothole detection: {str(e)}")
        return "None"

def analyze_image(image_path: str) -> Dict:
    """
    Analyze a single image for potholes
    
    Returns:
        dict: Detailed analysis results
    """
    try:
        if not os.path.exists(image_path):
            return {
                "potholes_detected": False,
                "detection_count": 0,
                "detections": [],
                "severity_levels": [],
                "confidence_scores": [],
                "error": "Image file not found"
            }
        
        logger.info(f"Analyzing image: {image_path}")
        filename = os.path.basename(image_path).lower()
        
        # Mock detection logic
        detections = []
        
        if "severe" in filename:
            detections = [
                {"severity": "Severe", "confidence": 0.92, "bbox": [100, 150, 300, 400]},
                {"severity": "Moderate", "confidence": 0.78, "bbox": [400, 200, 550, 350]}
            ]
        elif "moderate" in filename:
            detections = [
                {"severity": "Moderate", "confidence": 0.85, "bbox": [200, 180, 380, 320]}
            ]
        elif "multiple" in filename:
            detections = [
                {"severity": "Severe", "confidence": 0.89, "bbox": [50, 100, 200, 250]},
                {"severity": "Moderate", "confidence": 0.76, "bbox": [300, 150, 450, 280]},
                {"severity": "Moderate", "confidence": 0.82, "bbox": [500, 200, 650, 350]}
            ]
        elif "none" in filename or "good" in filename:
            detections = []
        else:
            # Random realistic detection
            if random.random() < 0.7:  # 70% chance of detection
                num_detections = random.choice([1, 1, 1, 2])  # Usually 1, sometimes 2
                for i in range(num_detections):
                    severity = random.choice(["Moderate", "Severe"])
                    confidence = random.uniform(0.6, 0.95)
                    bbox = [
                        random.randint(50, 300),
                        random.randint(100, 400),
                        random.randint(350, 600),
                        random.randint(450, 700)
                    ]
                    detections.append({
                        "severity": severity,
                        "confidence": confidence,
                        "bbox": bbox
                    })
        
        # Process results
        potholes_detected = len(detections) > 0
        severity_levels = [d["severity"] for d in detections]
        confidence_scores = [d["confidence"] for d in detections]
        
        result = {
            "potholes_detected": potholes_detected,
            "detection_count": len(detections),
            "detections": detections,
            "severity_levels": severity_levels,
            "confidence_scores": confidence_scores,
            "file_type": "image"
        }
        
        logger.info(f"Image analysis complete: {len(detections)} potholes detected")
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        return {
            "potholes_detected": False,
            "detection_count": 0,
            "detections": [],
            "severity_levels": [],
            "confidence_scores": [],
            "error": str(e)
        }

def analyze_video(video_path: str) -> Dict:
    """
    Analyze a video file for potholes
    
    Returns:
        dict: Detailed analysis results from video frames
    """
    try:
        if not os.path.exists(video_path):
            return {
                "potholes_detected": False,
                "detection_count": 0,
                "detections": [],
                "severity_levels": [],
                "confidence_scores": [],
                "frames_analyzed": 0,
                "error": "Video file not found"
            }
        
        logger.info(f"Analyzing video: {video_path}")
        filename = os.path.basename(video_path).lower()
        
        # Mock video analysis - simulate analyzing multiple frames
        frames_analyzed = random.randint(20, 60)  # Simulate frame extraction
        all_detections = []
        
        # Simulate finding potholes in various frames
        if "severe" in filename:
            # Multiple severe potholes across frames
            for frame_num in [5, 12, 18, 25, 31]:
                all_detections.append({
                    "frame": frame_num,
                    "timestamp": frame_num / 30.0,  # Assume 30 FPS
                    "severity": "Severe",
                    "confidence": random.uniform(0.85, 0.95),
                    "bbox": [
                        random.randint(100, 300),
                        random.randint(150, 350),
                        random.randint(400, 600),
                        random.randint(450, 650)
                    ]
                })
        elif "moderate" in filename:
            # A few moderate potholes
            for frame_num in [8, 22]:
                all_detections.append({
                    "frame": frame_num,
                    "timestamp": frame_num / 30.0,
                    "severity": "Moderate",
                    "confidence": random.uniform(0.70, 0.85),
                    "bbox": [
                        random.randint(150, 350),
                        random.randint(200, 400),
                        random.randint(450, 650),
                        random.randint(500, 700)
                    ]
                })
        elif "none" in filename or "good" in filename:
            all_detections = []
        else:
            # Random detections across frames
            detection_frames = random.sample(range(1, frames_analyzed), random.randint(1, 8))
            for frame_num in detection_frames:
                severity = random.choice(["Moderate", "Severe", "Moderate"])
                all_detections.append({
                    "frame": frame_num,
                    "timestamp": frame_num / 30.0,
                    "severity": severity,
                    "confidence": random.uniform(0.65, 0.90),
                    "bbox": [
                        random.randint(50, 400),
                        random.randint(100, 500),
                        random.randint(450, 750),
                        random.randint(550, 850)
                    ]
                })
        
        # Process results
        potholes_detected = len(all_detections) > 0
        severity_levels = [d["severity"] for d in all_detections]
        confidence_scores = [d["confidence"] for d in all_detections]
        
        # Get unique potholes (group nearby detections)
        unique_detections = _group_video_detections(all_detections)
        
        result = {
            "potholes_detected": potholes_detected,
            "detection_count": len(unique_detections),
            "detections": unique_detections,
            "severity_levels": severity_levels,
            "confidence_scores": confidence_scores,
            "frames_analyzed": frames_analyzed,
            "total_detections": len(all_detections),
            "file_type": "video"
        }
        
        logger.info(f"Video analysis complete: {len(unique_detections)} unique potholes detected across {frames_analyzed} frames")
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        return {
            "potholes_detected": False,
            "detection_count": 0,
            "detections": [],
            "severity_levels": [],
            "confidence_scores": [],
            "frames_analyzed": 0,
            "error": str(e)
        }

def _group_video_detections(detections: List[Dict]) -> List[Dict]:
    """
    Group video detections that likely represent the same pothole
    This is a simplified grouping based on severity and confidence
    """
    if not detections:
        return []
    
    # Simple grouping: take the best detection for each severity level
    grouped = {}
    
    for detection in detections:
        severity = detection["severity"]
        if severity not in grouped or detection["confidence"] > grouped[severity]["confidence"]:
            grouped[severity] = detection
    
    return list(grouped.values())

def analyze_media_file(file_path: str, file_type: str) -> Dict:
    """
    Main function to analyze either image or video file
    
    Args:
        file_path (str): Path to the media file
        file_type (str): "image" or "video"
        
    Returns:
        dict: Analysis results
    """
    try:
        logger.info(f"Starting {file_type} analysis: {file_path}")
        
        if file_type == "image":
            return analyze_image(file_path)
        elif file_type == "video":
            return analyze_video(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        logger.error(f"Error in media analysis: {str(e)}")
        return {
            "potholes_detected": False,
            "detection_count": 0,
            "detections": [],
            "severity_levels": [],
            "confidence_scores": [],
            "error": str(e)
        }

# Legacy compatibility function
def get_detection_details(image_path: str) -> dict:
    """Legacy function for backward compatibility"""
    return analyze_image(image_path)