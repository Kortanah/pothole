# database.py - Container optimized version
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import time
import logging

logger = logging.getLogger(__name__)

# Database URL - PostgreSQL for containers/Codespaces
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:password@localhost:5432/potholes"
)

def create_engine_with_retry(database_url, max_retries=5, retry_delay=2):
    """Create database engine with connection retry logic"""
    for attempt in range(max_retries):
        try:
            engine = create_engine(
                database_url,
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=300,    # Recycle connections every 5 minutes
                pool_size=5,         # Connection pool size
                max_overflow=10,     # Additional connections if needed
                echo=False           # Set to True for SQL debugging
            )
            
            # Test the connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            logger.info(f"✅ Database engine created successfully on attempt {attempt + 1}")
            return engine
            
        except Exception as e:
            logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("All database connection attempts failed")
                raise

# Create engine with retry logic
try:
    engine = create_engine_with_retry(DATABASE_URL)
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    # Create a dummy engine for development/testing
    engine = None

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None
Base = declarative_base()

def get_db():
    """Dependency to get database session"""
    if not SessionLocal:
        raise Exception("Database not available")
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create all tables"""
    if not engine:
        logger.error("Cannot create tables - database engine not available")
        return False
    
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        return False

def test_connection():
    """Test database connection"""
    if not engine:
        logger.error("Database engine not available")
        return False
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            logger.info(f"✅ PostgreSQL connection successful")
            logger.info(f"📋 PostgreSQL version: {version}")
            return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def wait_for_db(max_wait=30):
    """Wait for database to become available"""
    logger.info("⏳ Waiting for database to become available...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if test_connection():
            return True
        
        logger.info("Database not ready, waiting...")
        time.sleep(2)
    
    logger.error(f"Database did not become available within {max_wait} seconds")
    return False

# Container-specific helper functions
def start_postgresql_service():
    """Start PostgreSQL service in container environments"""
    import subprocess
    
    try:
        # Check if PostgreSQL is already running
        result = subprocess.run(
            ['pgrep', '-x', 'postgres'], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            logger.info("PostgreSQL is already running")
            return True
        
        # Try to start PostgreSQL service
        logger.info("Starting PostgreSQL service...")
        result = subprocess.run(
            ['sudo', 'service', 'postgresql', 'start'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("PostgreSQL service started successfully")
            # Wait a moment for service to fully start
            time.sleep(3)
            return True
        else:
            logger.error(f"Failed to start PostgreSQL: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error starting PostgreSQL service: {e}")
        return False

def ensure_database_ready():
    """Ensure database is ready for the application"""
    logger.info("🔧 Ensuring database is ready...")
    
    # Try to start PostgreSQL if needed
    if not start_postgresql_service():
        logger.warning("Could not start PostgreSQL service")
    
    # Wait for database to be available
    if wait_for_db():
        # Try to create tables
        return create_tables()
    
    return False