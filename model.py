# models.py
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship
    potholes = relationship("Pothole", back_populates="user")

class Pothole(Base):
    __tablename__ = "potholes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    image_path = Column(String)  # Path to image or video file
    media_type = Column(String, default="image")  # "image" or "video"
    latitude = Column(Float, default=0.0)
    longitude = Column(Float, default=0.0)
    gps_accuracy = Column(Float)  # GPS accuracy in meters
    capture_timestamp = Column(String)  # When the media was captured
    status = Column(String, default="Pending")  # Pending, In Progress, Fixed, Rejected
    severity = Column(String)  # None, Moderate, Severe
    confidence = Column(Float)  # AI confidence score
    detection_index = Column(Integer, default=0)  # For multiple detections in same media
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship
    user = relationship("User", back_populates="potholes")