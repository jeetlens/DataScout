
"""
DataScout FastAPI Backend
Main application entry point for the automated data analysis platform.
"""

import sys
import os

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="DataScout API",
    description="Automated Data Analysis & AI Insight Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# Include API routes
try:
    from api.routes import api_router
    app.include_router(api_router)
    logger.info("API routes loaded successfully")
except ImportError as e:
    logger.error(f"Failed to import API routes: {e}")

# Health check endpoint
@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint for health check."""
    return {"message": "DataScout API is running", "status": "healthy"}

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "DataScout API"}

# API versioning
@app.get("/api/v1/status")
async def api_status() -> Dict[str, Any]:
    """API status endpoint."""
    return {
        "api_version": "1.0.0",
        "status": "active",
        "features": {
            "data_loading": "enabled",
            "preprocessing": "enabled", 
            "ai_insights": "enabled",
            "report_generation": "enabled"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )