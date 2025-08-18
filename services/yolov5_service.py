# services/yolov5_service.py
import os
import sys
from pathlib import Path
import tempfile
import shutil
import cv2
from PIL import Image
import logging
from typing import Dict, Any, List, Optional
import torch
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import time

from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Project and directory setup
project_root = Path(__file__).resolve().parents[1]
YOLO_OUTPUT_DIR = "./runs/detect"
os.makedirs(YOLO_OUTPUT_DIR, exist_ok=True)

# Global variables for model state
models = {}  # Dictionary to store models for different devices
model_type = None
MODEL_LOADED = False
DEVICE_COUNT = 0
AVAILABLE_DEVICES = []

def confidence_to_severity(conf: float) -> str:
    """Map confidence score to severity level"""
    if conf > 0.85:
        return "high"
    elif conf > 0.6:
        return "medium"
    else:
        return "low"

def get_available_devices():
    """Get list of available devices (CPU + GPUs)"""
    devices = ['cpu']  # Always include CPU
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            devices.append(f'cuda:{i}')
        logger.info(f"Found {gpu_count} GPU(s) available")
    else:
        logger.info("No GPUs available, using CPU only")
    
    return devices

def load_yolo_model_on_device(device='cpu'):
    """Load YOLO model on specific device with optimized settings"""
    try:
        weights_path = project_root / 'best.pt'
        logger.info(f"Loading model on device: {device}")
        
        if not weights_path.exists():
            logger.error(f"Weights file not found: {weights_path}")
            return None, None
        
        # Skip ultralytics YOLO since we know it's incompatible
        # Go directly to torch hub loading for YOLOv5
        try:
            logger.info(f"Loading YOLOv5 model on {device} via torch hub...")
            
            # Load model with specific device and optimizations
            model = torch.hub.load(
                'ultralytics/yolov5', 
                'custom', 
                path=str(weights_path),
                device=device,
                trust_repo=True,
                force_reload=False
            )
            
            # CPU optimizations
            if device == 'cpu':
                model.cpu()
                # Enable CPU optimizations
                if hasattr(torch, 'set_num_threads'):
                    torch.set_num_threads(4)  # Adjust based on your CPU cores
                
                # Set inference optimizations
                model.conf = 0.25  # Confidence threshold
                model.iou = 0.45   # IoU threshold for NMS
                model.max_det = 1000  # Maximum detections per image
                
            # GPU optimizations
            else:
                model.to(device)
                # Enable half precision for GPU if supported
                if device.startswith('cuda'):
                    try:
                        model.half()  # Use FP16 for faster inference
                        logger.info(f"Enabled FP16 mode on {device}")
                    except:
                        logger.info(f"FP16 not supported on {device}, using FP32")
            
            model_type = 'yolov5_hub'
            logger.info(f"✅ Model loaded successfully on {device}")
            return model, model_type
            
        except Exception as e:
            logger.warning(f"Torch hub loading failed on {device}: {e}")
            return None, None
            
    except Exception as e:
        logger.error(f"❌ Error loading YOLO model on {device}: {e}")
        return None, None

def initialize_models():
    """Initialize YOLO models on all available devices"""
    global models, model_type, MODEL_LOADED, DEVICE_COUNT, AVAILABLE_DEVICES
    
    try:
        weights_path = project_root / 'best.pt'
        if not weights_path.exists():
            logger.warning(f"⚠️ Model file not found at {weights_path}")
            logger.warning("⚠️ Running in demo mode without detection capabilities")
            models = {}
            model_type = None
            MODEL_LOADED = False
            return
        
        AVAILABLE_DEVICES = get_available_devices()
        logger.info(f"🔄 Initializing YOLO models on devices: {AVAILABLE_DEVICES}")
        
        # Load models on available devices
        for device in AVAILABLE_DEVICES:
            logger.info(f"Loading model on {device}...")
            model, m_type = load_yolo_model_on_device(device)
            
            if model is not None:
                models[device] = model
                if model_type is None:
                    model_type = m_type
                logger.info(f"✅ Model loaded on {device}")
            else:
                logger.warning(f"❌ Failed to load model on {device}")
        
        MODEL_LOADED = len(models) > 0
        DEVICE_COUNT = len(models)
        
        if MODEL_LOADED:
            logger.info(f"✅ {DEVICE_COUNT} model(s) initialized successfully")
            logger.info(f"Available devices: {list(models.keys())}")
            
            # Test inference on primary device
            primary_device = list(models.keys())[0]
            test_model_inference(primary_device)
        else:
            logger.warning("⚠️ No models could be loaded - running in demo mode")
            
    except Exception as e:
        logger.error(f"❌ Error during model initialization: {e}")
        models = {}
        model_type = None
        MODEL_LOADED = False

def test_model_inference(device):
    """Test model inference to ensure it works"""
    try:
        model = models[device]
        
        # Create a test image
        test_img = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            test_img.save(tmp.name)
            
            # Try a test inference
            results = model(tmp.name)
            logger.info(f"✅ Model test inference successful on {device}")
            
            # Cleanup
            os.unlink(tmp.name)
            
    except Exception as test_error:
        logger.warning(f"⚠️ Model test inference failed on {device}: {test_error}")
        logger.warning("⚠️ Model loaded but may have issues with inference")

class ModelLoadBalancer:
    """Load balancer for distributing inference across multiple devices"""
    
    def __init__(self):
        self.device_queue = queue.Queue()
        self.device_usage = {device: 0 for device in models.keys()}
        
        # Initialize queue with available devices
        for device in models.keys():
            self.device_queue.put(device)
    
    def get_next_device(self):
        """Get next available device for inference"""
        if self.device_queue.empty():
            # If all devices are busy, wait for one to become available
            time.sleep(0.01)  # Small delay to prevent busy waiting
            return min(self.device_usage.keys(), key=self.device_usage.get)
        
        return self.device_queue.get()
    
    def release_device(self, device):
        """Release device back to the pool"""
        self.device_usage[device] -= 1
        self.device_queue.put(device)
    
    def acquire_device(self, device):
        """Mark device as in use"""
        self.device_usage[device] += 1

# Global load balancer instance
load_balancer = ModelLoadBalancer() if MODEL_LOADED else None

def detect_pothole_yolov5_hub(image_path: str, model, device: str) -> Dict[str, Any]:
    """Detect potholes using YOLOv5 hub model with device specification"""
    try:
        logger.debug(f"Running inference on {device} for {image_path}")
        
        # Run inference
        results = model(image_path)
        
        detections = []
        confidences = []
        severities = []
        
        # Parse results
        if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
            for *xyxy, conf, cls in results.xyxy[0].tolist():
                detections.append({
                    'bbox': xyxy,
                    'confidence': float(conf),
                    'class_id': int(cls)
                })
                confidences.append(float(conf))
                severities.append(confidence_to_severity(float(conf)))
        
        # Save annotated image
        try:
            results.save(save_dir=YOLO_OUTPUT_DIR)
        except Exception as e:
            logger.warning(f"Could not save annotated image: {e}")
        
        return {
            "success": True,
            "detections": detections,
            "confidences": confidences,
            "severities": severities,
            "device_used": device
        }
        
    except Exception as e:
        logger.error(f"Error in YOLOv5 hub model inference on {device}: {e}")
        return {
            "success": False,
            "error": str(e),
            "detections": [],
            "confidences": [],
            "severities": [],
            "device_used": device
        }

def run_detection_single(image_path: str, device: str = None) -> Dict[str, Any]:
    """Run detection on a single device"""
    if not MODEL_LOADED:
        return {
            "success": False,
            "error": "Model not loaded",
            "detections": [],
            "confidences": [],
            "severities": []
        }
    
    # Select device
    if device is None or device not in models:
        device = list(models.keys())[0]  # Use first available device
    
    try:
        model = models[device]
        return detect_pothole_yolov5_hub(image_path, model, device)
        
    except Exception as e:
        logger.error(f"Error in run_detection_single on {device}: {e}")
        return {
            "success": False,
            "error": str(e),
            "detections": [],
            "confidences": [],
            "severities": []
        }

def run_detection_multi_threaded(image_paths: List[str]) -> List[Dict[str, Any]]:
    """Run detection on multiple images using multiple devices with threading"""
    if not MODEL_LOADED:
        return [{
            "success": False,
            "error": "Model not loaded",
            "detections": [],
            "confidences": [],
            "severities": [],
            "image_path": path
        } for path in image_paths]
    
    if len(models) == 1:
        # Single device - process sequentially
        results = []
        device = list(models.keys())[0]
        for image_path in image_paths:
            result = run_detection_single(image_path, device)
            result["image_path"] = image_path
            results.append(result)
        return results
    
    # Multi-device processing with threading
    results = []
    
    def worker_function(image_path: str) -> Dict[str, Any]:
        """Worker function for threaded inference"""
        # Get next available device
        device = load_balancer.get_next_device()
        load_balancer.acquire_device(device)
        
        try:
            result = run_detection_single(image_path, device)
            result["image_path"] = image_path
            return result
        finally:
            load_balancer.release_device(device)
    
    # Use ThreadPoolExecutor for parallel processing
    max_workers = min(len(models), len(image_paths))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(worker_function, path): path 
                         for path in image_paths}
        
        for future in as_completed(future_to_path):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                path = future_to_path[future]
                logger.error(f"Error processing {path}: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "detections": [],
                    "confidences": [],
                    "severities": [],
                    "image_path": path
                })
    
    # Sort results to match input order
    path_to_result = {r["image_path"]: r for r in results}
    return [path_to_result[path] for path in image_paths]

def run_detection(image_path: str) -> Dict[str, Any]:
    """Run detection using load balancing across available devices"""
    if isinstance(image_path, list):
        # Multiple images
        return run_detection_multi_threaded(image_path)
    else:
        # Single image
        return run_detection_single(image_path)

def detect_pothole(image_path: str) -> bool:
    """Quick check if pothole exists in an image"""
    if not MODEL_LOADED:
        logger.warning("Model not loaded, cannot detect potholes")
        return False

    try:
        result = run_detection(image_path)
        return result["success"] and len(result["detections"]) > 0
    except Exception as e:
        logger.error(f"Error in detection: {e}")
        return False

def analyze_media_file(file_path: str, media_type: str) -> Dict[str, Any]:
    """Analyze image or video for potholes"""
    if not os.path.exists(file_path):
        return {
            "potholes_detected": False,
            "detection_count": 0,
            "detections": [],
            "severity_levels": [],
            "confidence_scores": [],
            "media_type": media_type,
            "file_path": file_path,
            "error": f"File not found: {file_path}"
        }

    try:
        # For video files, extract a frame first
        analysis_path = file_path
        if media_type == "video":
            analysis_path = extract_video_frame(file_path)
            if not analysis_path:
                return {
                    "potholes_detected": False,
                    "detection_count": 0,
                    "detections": [],
                    "severity_levels": [],
                    "confidence_scores": [],
                    "media_type": media_type,
                    "file_path": file_path,
                    "error": "Could not extract frame from video"
                }

        # Run detection
        result = run_detection(analysis_path)
        
        if not result["success"]:
            return {
                "potholes_detected": False,
                "detection_count": 0,
                "detections": [],
                "severity_levels": [],
                "confidence_scores": [],
                "media_type": media_type,
                "file_path": file_path,
                "error": result.get("error", "Detection failed")
            }

        detections = result["detections"]
        confidences = result["confidence_scores"] if "confidence_scores" in result else result["confidences"]
        severities = result["severity_levels"] if "severity_levels" in result else result["severities"]
        num_detections = len(detections)

        # Get the path to the labeled image
        labeled_image_url = None
        try:
            exp_dirs = [d for d in os.listdir(YOLO_OUTPUT_DIR) if os.path.isdir(os.path.join(YOLO_OUTPUT_DIR, d))]
            if exp_dirs:
                latest_exp = sorted(exp_dirs, key=lambda x: os.path.getmtime(os.path.join(YOLO_OUTPUT_DIR, x)))[-1]
                labeled_image_url = f"/runs/detect/{latest_exp}/{Path(file_path).name}"
        except Exception as e:
            logger.warning(f"Could not determine labeled image URL: {e}")

        return {
            "potholes_detected": num_detections > 0,
            "detection_count": num_detections,
            "detections": detections,
            "severity_levels": severities,
            "confidence_scores": confidences,
            "labeled_image_url": labeled_image_url,
            "model_type": model_type,
            "media_type": media_type,
            "file_path": file_path,
            "device_used": result.get("device_used", "unknown")
        }

    except Exception as e:
        logger.error(f"Error analyzing {file_path}: {e}")
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

def analyze_batch_images(image_paths: List[str]) -> List[Dict[str, Any]]:
    """Analyze multiple images in parallel using multi-device inference"""
    if not MODEL_LOADED:
        return [{
            "potholes_detected": False,
            "detection_count": 0,
            "detections": [],
            "severity_levels": [],
            "confidence_scores": [],
            "media_type": "image",
            "file_path": path,
            "error": "Model not loaded"
        } for path in image_paths]
    
    logger.info(f"Analyzing batch of {len(image_paths)} images using {len(models)} device(s)")
    
    # Filter existing files
    existing_paths = [path for path in image_paths if os.path.exists(path)]
    if len(existing_paths) != len(image_paths):
        logger.warning(f"Some files not found. Processing {len(existing_paths)}/{len(image_paths)} files")
    
    # Run batch detection
    detection_results = run_detection_multi_threaded(existing_paths)
    
    # Process results
    final_results = []
    for result in detection_results:
        file_path = result["image_path"]
        
        if not result["success"]:
            final_results.append({
                "potholes_detected": False,
                "detection_count": 0,
                "detections": [],
                "severity_levels": [],
                "confidence_scores": [],
                "media_type": "image",
                "file_path": file_path,
                "error": result.get("error", "Detection failed"),
                "device_used": result.get("device_used", "unknown")
            })
            continue

        detections = result["detections"]
        confidences = result["confidences"]
        severities = result["severities"]
        num_detections = len(detections)

        # Get labeled image URL
        labeled_image_url = None
        try:
            exp_dirs = [d for d in os.listdir(YOLO_OUTPUT_DIR) if os.path.isdir(os.path.join(YOLO_OUTPUT_DIR, d))]
            if exp_dirs:
                latest_exp = sorted(exp_dirs, key=lambda x: os.path.getmtime(os.path.join(YOLO_OUTPUT_DIR, x)))[-1]
                labeled_image_url = f"/runs/detect/{latest_exp}/{Path(file_path).name}"
        except Exception as e:
            logger.warning(f"Could not determine labeled image URL for {file_path}: {e}")

        final_results.append({
            "potholes_detected": num_detections > 0,
            "detection_count": num_detections,
            "detections": detections,
            "severity_levels": severities,
            "confidence_scores": confidences,
            "labeled_image_url": labeled_image_url,
            "model_type": model_type,
            "media_type": "image",
            "file_path": file_path,
            "device_used": result.get("device_used", "unknown")
        })
    
    return final_results

def extract_video_frame(video_path: str) -> Optional[str]:
    """Extract a frame from video for analysis"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Get middle frame
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame = frame_count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        # Save frame as temporary image
        temp_path = video_path.replace('.', '_frame.') 
        if temp_path.split('.')[-1].lower() in ['mp4', 'avi', 'mov', 'mkv']:
            temp_path = temp_path.rsplit('.', 1)[0] + '.jpg'
        
        cv2.imwrite(temp_path, frame)
        return temp_path
        
    except Exception as e:
        logger.error(f"Error extracting frame from video: {e}")
        return None

def get_model_status() -> Dict[str, Any]:
    """Get current model status for debugging"""
    weights_path = project_root / 'best.pt'
    yolo_path = project_root / 'yolov5'
    
    return {
        "model_loaded": MODEL_LOADED,
        "model_type": model_type if MODEL_LOADED else None,
        "device_count": DEVICE_COUNT,
        "available_devices": AVAILABLE_DEVICES,
        "loaded_devices": list(models.keys()) if models else [],
        "yolo_path_exists": yolo_path.exists(),
        "yolo_path": str(yolo_path),
        "weights_path": str(weights_path),
        "weights_exists": weights_path.exists(),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "python_path": sys.path[:5]  # First 5 entries
    }

def get_inference_stats() -> Dict[str, Any]:
    """Get inference statistics"""
    if load_balancer:
        return {
            "device_usage": load_balancer.device_usage.copy(),
            "queue_size": load_balancer.device_queue.qsize()
        }
    return {"device_usage": {}, "queue_size": 0}

# Initialize models when module is imported
initialize_models()

# Reinitialize load balancer after models are loaded
if MODEL_LOADED:
    load_balancer = ModelLoadBalancer()