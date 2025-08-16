# services/__init__.py
"""
Services package for pothole detection application
"""

from .yolov5_service import detect_pothole, get_detection_details

__all__ = ['detect_pothole', 'get_detection_details']