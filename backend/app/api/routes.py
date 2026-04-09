import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any
from app.services.search_service import search_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure router
router = APIRouter(prefix="/api", tags=["search"])
class TextSearchRequest(BaseModel):
    """Request model for text search"""
    query: str
class SearchResponse(BaseModel):
    """Response model for search results"""
    results: List[Dict[str, Any]]
    count: int

@router.post("/search/text", response_model=SearchResponse)
async def search_text(request: TextSearchRequest):
    """
    Search using text query
    Args:
        request: Text search request with query
    Returns:
        Search results with video_id, frame_idx, image_path, score
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        results = search_service.search_by_text(request.query)
        
        return SearchResponse(
            results=results,
            count=len(results)
        )
        
    except Exception as e:
        import traceback
        logger.error(f"Text search failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    from app.database.milvus_client import milvus_client
    
    try:
        stats = milvus_client.get_stats()
        
        return {
            "status": "healthy",
            "milvus": "connected",
            "collection": stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }