-- init.sql - Initialize database schema
-- This file will be executed when PostgreSQL container starts for the first time

-- Create database if it doesn't exist (PostgreSQL will handle this via POSTGRES_DB env var)

-- Create tables (these will also be created by SQLAlchemy, but having them here ensures consistency)
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS potholes (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    image_path VARCHAR(500),
    media_type VARCHAR(10) DEFAULT 'image',
    latitude FLOAT DEFAULT 0.0,
    longitude FLOAT DEFAULT 0.0,
    gps_accuracy FLOAT,
    capture_timestamp VARCHAR(255),
    status VARCHAR(50) DEFAULT 'Pending',
    severity VARCHAR(50),
    confidence FLOAT,
    detection_index INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_potholes_user_id ON potholes(user_id);
CREATE INDEX IF NOT EXISTS idx_potholes_status ON potholes(status);
CREATE INDEX IF NOT EXISTS idx_potholes_created_at ON potholes(created_at);
CREATE INDEX IF NOT EXISTS idx_potholes_location ON potholes(latitude, longitude);

-- Create a default admin user (password: admin123)
-- Note: This is hashed with bcrypt
INSERT INTO users (username, email, hashed_password, is_admin) 
VALUES (
    'admin', 
    'admin@potholeapp.com', 
    '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW',  -- admin123
    TRUE
) ON CONFLICT (username) DO NOTHING;

-- Create a test user (password: testuser123)  
INSERT INTO users (username, email, hashed_password, is_admin)
VALUES (
    'testuser',
    'test@potholeapp.com',
    '$2b$12$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi',  -- testuser123
    FALSE
) ON CONFLICT (username) DO NOTHING;