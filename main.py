# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routers import auth, users, potholes
from database import create_tables, test_connection
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Pothole Detection API",
    description="API for reporting and managing potholes using AI detection",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
REQUIRED_DIRS = ["uploads", "analysis", "captures", "services", "runs/detect", "yolov5"]
for directory in REQUIRED_DIRS:
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Ensured directory exists: {directory}")

# Mount static files for serving uploaded images and YOLO results
if os.path.exists("uploads"):
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Mount analysis directory to serve analysis mode uploads
if os.path.exists("analysis"):
    app.mount("/analysis", StaticFiles(directory="analysis"), name="analysis")
    logger.info("Mounted /analysis directory for uploaded file serving")

# Mount captures directory to serve live capture uploads
if os.path.exists("captures"):
    app.mount("/captures", StaticFiles(directory="captures"), name="captures")
    logger.info("Mounted /captures directory for captured file serving")

# Mount static files to serve YOLO detection images (same as your original code)
if os.path.exists("runs"):
    app.mount("/runs", StaticFiles(directory="runs"), name="runs")
    logger.info("Mounted /runs directory for YOLO output serving")

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting Pothole Detection API...")
    
    # Check for required model files
    model_path = "./best.pt"
    yolov5_path = "./yolov5"
    
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found at {model_path}")
        logger.warning("Please ensure you have your trained YOLOv5 model at ./best.pt")
    
    if not os.path.exists(yolov5_path):
        logger.warning(f"YOLOv5 directory not found at {yolov5_path}")
        logger.warning("Please ensure you have YOLOv5 code in ./yolov5 directory")
    
    # Test database connection
    if not test_connection():
        logger.error("Database connection failed! Check your PostgreSQL setup.")
        # Don't raise exception in development - allow app to start for testing
        logger.warning("Continuing startup despite database issues...")
    else:
        logger.info("Database connection successful")
        
        # Create tables
        try:
            create_tables()
            logger.info("Database tables created/verified successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
    
    logger.info("Startup completed successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down Pothole Detection API...")

# Include routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(potholes.router)

@app.get("/")
def read_root():
    return {
        "message": "Pothole Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "OK",
        "yolo_model_loaded": os.path.exists("./best.pt"),
        "yolo_code_available": os.path.exists("./yolov5")
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db_status = test_connection()
        model_status = os.path.exists("./best.pt")
        yolo_status = os.path.exists("./yolov5")
        
        return {
            "status": "healthy" if (db_status and model_status and yolo_status) else "degraded",
            "service": "pothole-detection-api",
            "database": "connected" if db_status else "disconnected",
            "yolo_model": "loaded" if model_status else "missing",
            "yolo_code": "available" if yolo_status else "missing",
            "version": "1.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "pothole-detection-api",
            "error": str(e),
            "version": "1.0.0"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )