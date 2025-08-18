# Use Python 3.9 slim for smaller image size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    libomp-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libsndfile1 \
    ffmpeg \
    curl \
    git \
    gcc \
    g++ \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pre-create directories
RUN mkdir -p uploads analysis captures services runs/detect yolov5 \
    && chmod -R 777 /app

# Copy requirements first
COPY requirements.txt .

# Upgrade pip and wheel
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU build) and dependencies
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install \
    numpy==1.23.5 \
    opencv-python-headless==4.8.0.74 \
    Pillow==10.0.0 \
    psutil==5.9.5 \
    matplotlib==3.7.1 \
    PyYAML>=5.3.1 \
    requests>=2.23.0 \
    scipy>=1.10.1 \
    tqdm>=4.66.3 \
    ultralytics==8.2.64 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    sqlalchemy==2.0.23 \
    pydantic[email]==2.5.0 \
    python-multipart==0.0.6 \
    python-jose[cryptography]==3.3.0 \
    passlib[bcrypt]==1.7.4 \
    psycopg2-binary==2.9.9 \
    alembic==1.12.1

# Copy application code
COPY . .

# Explicitly copy best.pt into the image
COPY best.pt /app/best.pt

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /home/appuser/.cache/torch/hub \
    && mkdir -p /home/appuser/.config/Ultralytics \
    && chown -R appuser:appuser /app /home/appuser

# Switch to non-root
USER appuser

# Expose port and env vars
EXPOSE 8000
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
