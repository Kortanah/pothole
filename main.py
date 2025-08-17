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
REQUIRED_DIRS = ["uploads", "analysis", "captures", "services"]
for directory in REQUIRED_DIRS:
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Ensured directory exists: {directory}")

# Mount static files for serving uploaded images
if os.path.exists("uploads"):
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting Pothole Detection API...")
    
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
        "health": "OK"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db_status = test_connection()
        return {
            "status": "healthy" if db_status else "degraded",
            "service": "pothole-detection-api",
            "database": "connected" if db_status else "disconnected",
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