import logging
import numpy as np
from pymilvus import AnnSearchRequest, WeightedRanker
from typing import List, Dict, Any
from app.config import settings
from app.database.milvus_client import milvus_client
from app.models.model_manager import model_manager
from app.services.translation_service import translation_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SearchService:
    """Service for hybrid search with reranking"""
    
    @staticmethod
    def search_by_text(query: str) -> List[Dict[str, Any]]:
        """
        Search using text query with hybrid strategy
        Args:
            query: Text search query  
        Returns:
            List of search results with metadata and scores
        """
        try:
            logger.info(f"Searching for text query: {query}")
            
            # Translate query to English for CLIP models (Keyframe + Object)
            query_en = translation_service.translate_vi_to_en(query)
            
            # Generate embeddings for all three modalities
            # Use CLIP for keyframe and object (English query), OCR model for text (Original query)
            kf_embedding = model_manager.encode_text(query_en, use_ocr_model=False)
            ocr_embedding = model_manager.encode_text(query, use_ocr_model=True)
            obj_embedding = model_manager.encode_text(query_en, use_ocr_model=False)
            
            # Perform hybrid search
            results = SearchService._hybrid_search(kf_embedding=kf_embedding, ocr_embedding=ocr_embedding, obj_embedding=obj_embedding)
            
            # Rerank using cross-encoder
            results = SearchService._rerank_results(query, results)
            
            logger.info(f"Found {len(results)} results")
            return results[:settings.final_top_k]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    @staticmethod
    def search_by_embeddings(kf_embedding: np.ndarray, ocr_embedding: np.ndarray, obj_embedding: np.ndarray, query_text: str = "") -> List[Dict[str, Any]]:
        """
        Search using precomputed embeddings (for video queries)
        Args:
            kf_embedding: Keyframe embedding
            ocr_embedding: OCR embedding
            obj_embedding: Object embedding
            query_text: Optional text for reranking
        Returns:
            List of search results
        """
        try:
            # Perform hybrid search
            results = SearchService._hybrid_search(
                kf_embedding=kf_embedding,
                ocr_embedding=ocr_embedding,
                obj_embedding=obj_embedding
            )
            
            # Rerank if query text provided
            if query_text:
                results = SearchService._rerank_results(query_text, results)
            
            return results[:settings.final_top_k]
            
        except Exception as e:
            logger.error(f"Embedding search failed: {e}")
            raise
    
    @staticmethod
    def _hybrid_search(kf_embedding: np.ndarray, ocr_embedding: np.ndarray, obj_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using Milvus WeightedRanker
        Args:
            kf_embedding: Keyframe embedding vector
            ocr_embedding: OCR embedding vector
            obj_embedding: Object embedding vector
        Returns:
            List of candidate results from Milvus
        """
        try:
            collection = milvus_client.get_collection()
            
            # Create ANN search requests for each vector field
            search_param = {
                "metric_type": "IP",
                "params": {"nprobe": 64}
            }

            # Keyframe search parameters
            kf_param = {
                'data': [kf_embedding.tolist()],
                'anns_field': 'kf_embedding',
                'param': search_param,
                'limit': settings.top_k
            }
            
            # Keyframe search request
            kf_request = AnnSearchRequest(**kf_param)
            
            # OCR search parameters
            ocr_param = {
                'data': [ocr_embedding.tolist()],
                'anns_field': 'ocr_embedding',
                'param': search_param,
                'limit': settings.top_k
            }

            # OCR search request
            ocr_request = AnnSearchRequest(**ocr_param)

            # Object search parameters
            obj_param = {
                'data': [obj_embedding.tolist()],
                'anns_field': 'obj_embedding',
                'param': search_param,
                'limit': settings.top_k
            }
            
            # Object search request
            obj_request = AnnSearchRequest(**obj_param)

            reqs = [kf_request, ocr_request, obj_request]
            
            # Create weighted ranker
            ranker = WeightedRanker(
                settings.kf_weight,
                settings.ocr_weight,
                settings.obj_weight
            )
            
            # Perform hybrid search
            search_results = collection.hybrid_search(
                reqs=reqs,
                rerank=ranker,
                limit=settings.top_k,
                output_fields=["video_id", "frame_idx", "image_path", "ocr_text"]
            )
            
            # Parse results
            results = []
            for hits in search_results:
                for hit in hits:
                    results.append({
                        "video_id": hit.get("video_id"),
                        "frame_idx": hit.get("frame_idx"),
                        "image_path": hit.get("image_path"),
                        "ocr_text": hit.get("ocr_text", ""),
                        "score": float(hit.score)
                    })
            
            logger.info(f"Hybrid search returned {len(results)} candidates")
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise
    
    @staticmethod
    def _rerank_results(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank results using cross-encoder
        
        Args:
            query: Search query
            results: Initial search results
            
        Returns:
            Reranked results
        """
        try:
            if not results:
                return results
            
            # Take top N for reranking to save time
            candidates = results[:settings.rerank_top_k]
            
            # Prepare candidates for reranking
            candidate_pairs = []
            
            # Map context to original result for reconstruction
            context_to_result = {}
            
            for result in candidates:
                context = f"{result['ocr_text']}" if result['ocr_text'] else 'Empty OCR text'
                context_to_result[context] = result
                candidate_pairs.append((context, result['score']))
            
            # rerank() returns sorted list of (context, new_score)
            reranked_pairs = model_manager.rerank(query, candidate_pairs)
            
            # Reconstruct results based on reranked order
            reranked_results = []
            for context, new_score in reranked_pairs:
                if context in context_to_result:
                    original_result = context_to_result[context]
                    original_result['score'] = new_score
                    reranked_results.append(original_result)
            
            # Append remaining results without reranking
            reranked_results.extend(results[settings.rerank_top_k:])
            
            logger.info(f"Reranked top {len(reranked_pairs)} results")
            return reranked_results
            
        except Exception as e:
            logger.warning(f"Reranking failed, returning original results: {e}")
            return results

search_service = SearchService()