import os
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings
from app.database.milvus_client import milvus_client
from app.models.model_manager import model_manager
from app.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("Starting Video Search API...")
    
    try:
        # Connect to Milvus
        milvus_client.connect()
        
        # Initialize models (lazy loading happens on first use)
        logger.info("Model manager initialized")
        
        logger.info("Application startup complete")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    milvus_client.disconnect()
    model_manager.clear_cache()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.isdir(settings.data_dir):
    app.mount("/static", StaticFiles(directory=settings.data_dir), name="static")
    logger.info(f"Mounted static files from {settings.data_dir} at /static")
else:
    logger.warning(f"Static directory not found: {settings.data_dir}")

# Include routers
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Video Search API",
        "version": settings.api_version,
        "docs": "/docs"
    }