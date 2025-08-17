# services/yolov5_service.py
import os
import sys
from pathlib import Path
import tempfile
import shutil

# Fix utils import conflict by temporarily renaming our utils
project_root = Path(__file__).resolve().parents[1]
utils_path = project_root / 'utils'
temp_utils_path = None

def setup_yolo_environment():
    """Set up environment for YOLOv5 loading"""
    global temp_utils_path
    
    # Temporarily rename our utils directory to avoid conflicts
    if utils_path.exists():
        temp_utils_path = project_root / 'utils_backup'
        if temp_utils_path.exists():
            shutil.rmtree(temp_utils_path)
        shutil.move(str(utils_path), str(temp_utils_path))
    
    # Add YOLOv5 to Python path
    yolo_path = str(project_root / 'yolov5')
    if yolo_path not in sys.path:
        sys.path.insert(0, yolo_path)
    
    # Remove any cached utils modules
    modules_to_remove = [k for k in sys.modules.keys() if k.startswith('utils')]
    for module in modules_to_remove:
        del sys.modules[module]

def restore_environment():
    """Restore original environment"""
    global temp_utils_path
    
    # Restore our utils directory
    if temp_utils_path and temp_utils_path.exists():
        if utils_path.exists():
            shutil.rmtree(utils_path)
        shutil.move(str(temp_utils_path), str(utils_path))
        temp_utils_path = None

import cv2
from PIL import Image
import logging
from typing import Dict, Any
import torch
import numpy as np

logger = logging.getLogger(__name__)

# Directory for YOLO output
YOLO_OUTPUT_DIR = "./runs/detect"
os.makedirs(YOLO_OUTPUT_DIR, exist_ok=True)


def load_yolo_model():
    """Load YOLO model with proper environment setup"""
    try:
        weights_path = project_root / 'best.pt'
        
        logger.info(f"Loading model from {weights_path}")
        
        # Check if weights file exists
        if not weights_path.exists():
            logger.error(f"Weights file not found: {weights_path}")
            return None, None
        
        # Method 1: Try direct PyTorch loading with proper settings
        try:
            logger.info("Attempting direct PyTorch loading...")
            
            # Allow numpy functions needed for YOLOv5
            import torch.serialization
            safe_globals = [
                'numpy.core.multiarray._reconstruct',
                'numpy.ndarray',
                'numpy.dtype',
                'numpy.core.multiarray.scalar',
                'collections.OrderedDict',
                '__builtin__.object'
            ]
            
            # Load with weights_only=False for compatibility
            with torch.serialization.safe_globals(safe_globals):
                if torch.cuda.is_available():
                    checkpoint = torch.load(weights_path, map_location='cuda', weights_only=False)
                else:
                    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            
            # Extract model from checkpoint
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model = checkpoint['model'].float()
                elif 'ema' in checkpoint:
                    model = checkpoint['ema'].float()
                else:
                    # Try to find the model in the checkpoint
                    for key in checkpoint.keys():
                        if hasattr(checkpoint[key], 'forward'):
                            model = checkpoint[key].float()
                            break
                    else:
                        raise ValueError("Could not find model in checkpoint")
            else:
                model = checkpoint.float()
            
            model.eval()
            logger.info("✅ YOLO model loaded successfully using direct PyTorch loading")
            return model, 'pytorch'
            
        except Exception as e:
            logger.warning(f"Direct PyTorch loading failed: {e}")
        
        # Method 2: Try with environment setup for YOLOv5 hub loading
        try:
            logger.info("Setting up YOLOv5 environment...")
            setup_yolo_environment()
            
            try:
                # Try loading from ultralytics/yolov5
                model = torch.hub.load(
                    'ultralytics/yolov5', 
                    'custom', 
                    path=str(weights_path),
                    trust_repo=True,
                    force_reload=True
                )
                logger.info("✅ YOLO model loaded successfully using ultralytics/yolov5")
                return model, 'yolov5'
                
            except Exception as e:
                logger.warning(f"Failed ultralytics/yolov5 loading: {e}")
                
                # Try loading from local YOLOv5
                yolo_dir = str(project_root / 'yolov5')
                if os.path.exists(yolo_dir):
                    original_cwd = os.getcwd()
                    try:
                        os.chdir(project_root)
                        model = torch.hub.load(
                            yolo_dir,
                            'custom',
                            path=str(weights_path),
                            source='local',
                            trust_repo=True,
                            force_reload=True
                        )
                        logger.info("✅ YOLO model loaded successfully using local yolov5")
                        return model, 'yolov5'
                    finally:
                        os.chdir(original_cwd)
                        
        except Exception as e:
            logger.warning(f"Hub loading failed: {e}")
        
        finally:
            restore_environment()
        
        logger.error("❌ All model loading methods failed")
        return None, None
            
    except Exception as e:
        logger.error(f"❌ Unexpected error loading YOLO model: {e}")
        restore_environment()
        return None, None


def confidence_to_severity(conf: float) -> str:
    """Map confidence score to severity level"""
    if conf > 0.85:
        return "high"
    elif conf > 0.6:
        return "medium"
    else:
        return "low"


def detect_pothole_with_pytorch_model(image_path: str, model) -> Dict[str, Any]:
    """Detect potholes using direct PyTorch model"""
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (typically 640x640 for YOLOv5)
        img_resized = cv2.resize(img_rgb, (640, 640))
        
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
            model = model.cuda()
        
        # Run inference
        with torch.no_grad():
            predictions = model(img_tensor)
        
        # Process predictions (this is a simplified version)
        # You might need to adjust this based on your model's output format
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        
        detections = []
        confidences = []
        severities = []
        
        # Basic post-processing (you may need to adjust this)
        if hasattr(predictions, 'cpu'):
            pred_cpu = predictions.cpu().numpy()
            
            # This is a simplified parsing - adjust based on your model's output format
            if len(pred_cpu.shape) == 3:  # [batch, detections, attrs]
                for detection in pred_cpu[0]:  # First batch item
                    if len(detection) >= 6:  # x1, y1, x2, y2, conf, class
                        conf = float(detection[4])
                        if conf > 0.3:  # Confidence threshold
                            detections.append({
                                'bbox': detection[:4].tolist(),
                                'confidence': conf,
                                'class_id': int(detection[5]) if len(detection) > 5 else 0
                            })
                            confidences.append(conf)
                            severities.append(confidence_to_severity(conf))
        
        return {
            "success": True,
            "detections": detections,
            "confidences": confidences,
            "severities": severities
        }
        
    except Exception as e:
        logger.error(f"Error in PyTorch model inference: {e}")
        return {
            "success": False,
            "error": str(e),
            "detections": [],
            "confidences": [],
            "severities": []
        }


def detect_pothole_with_yolov5_model(image_path: str, model) -> Dict[str, Any]:
    """Detect potholes using YOLOv5 hub model"""
    try:
        # Use YOLOv5 model inference
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
            "severities": severities
        }
        
    except Exception as e:
        logger.error(f"Error in YOLOv5 model inference: {e}")
        return {
            "success": False,
            "error": str(e),
            "detections": [],
            "confidences": [],
            "severities": []
        }


# Try to load model at startup
try:
    model, model_type = load_yolo_model()
    MODEL_LOADED = model is not None
    if MODEL_LOADED:
        logger.info(f"✅ Model loaded successfully with type: {model_type}")
    else:
        logger.warning("⚠️ No model could be loaded")
except Exception as e:
    logger.error(f"❌ Error during model loading: {e}")
    model, model_type = None, None
    MODEL_LOADED = False


def detect_pothole(image_path: str) -> bool:
    """Quick check if pothole exists in an image"""
    if not MODEL_LOADED:
        logger.warning("Model not loaded, cannot detect potholes")
        return False

    try:
        if model_type == 'yolov5':
            result = detect_pothole_with_yolov5_model(image_path, model)
        else:
            result = detect_pothole_with_pytorch_model(image_path, model)
        
        return result["success"] and len(result["detections"]) > 0
    except Exception as e:
        logger.error(f"Error in detection: {e}")
        return False


def analyze_media_file(file_path: str, media_type: str) -> Dict[str, Any]:
    """Analyze image or video for potholes"""
    if not MODEL_LOADED:
        logger.error("Model not loaded")
        return {
            "potholes_detected": False,
            "detection_count": 0,
            "detections": [],
            "severity_levels": [],
            "confidence_scores": [],
            "media_type": media_type,
            "file_path": file_path,
            "error": "Model not loaded"
        }

    try:
        # Check if file exists
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

        # Use appropriate detection method based on model type
        if model_type == 'yolov5':
            result = detect_pothole_with_yolov5_model(file_path, model)
        else:
            result = detect_pothole_with_pytorch_model(file_path, model)
        
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
        confidences = result["confidences"]
        severities = result["severities"]
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
            "file_path": file_path
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


def get_model_status() -> Dict[str, Any]:
    """Get current model status for debugging"""
    weights_path = project_root / 'best.pt'
    yolo_path = project_root / 'yolov5'
    
    return {
        "model_loaded": MODEL_LOADED,
        "model_type": model_type if MODEL_LOADED else None,
        "yolo_path_exists": yolo_path.exists(),
        "yolo_path": str(yolo_path),
        "weights_path": str(weights_path),
        "weights_exists": weights_path.exists(),
        "utils_conflict_resolved": temp_utils_path is None,
        "torch_cuda_available": torch.cuda.is_available(),
        "python_path": sys.path[:5]  # First 5 entries
    }