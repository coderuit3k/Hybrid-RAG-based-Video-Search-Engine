import numpy as np
import logging
import torch
import open_clip
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
from FlagEmbedding import FlagReranker
from PIL import Image
from typing import List, Union, Tuple
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton class for managing embedding models with memory optimization"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        logger.info("Initializing ModelManager...")
        self.device = settings.model_device
        self.reranker_device = settings.reranker_device
        
        # Initialize models
        self._load_clip_model()
        self._load_ocr_model()
        self._load_reranker()
        
        self._initialized = True
        logger.info(f"ModelManager initialized. Device: {self.device}, Reranker: {self.reranker_device}")
    
    def _load_clip_model(self):
        """Load CLIP model for keyframe and object embeddings"""
        try:
            logger.info(f"Loading CLIP model: {settings.clip_model}")
            self.clip_processor = CLIPProcessor.from_pretrained(settings.clip_model)
            self.clip_model = CLIPModel.from_pretrained(settings.clip_model).to(self.device)
            
            self.clip_model.eval()
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def _load_ocr_model(self):
        """Load XLM-Roberta CLIP variant for OCR embeddings"""
        try:
            logger.info(f"Loading OCR model: {settings.ocr_model}")
            # Use open_clip for multilingual support
            self.ocr_model, _, self.ocr_preprocess = open_clip.create_model_and_transforms(
                settings.ocr_model,
                pretrained=settings.ocr_pretrained,
                device=self.device
            )
            self.ocr_tokenizer = open_clip.get_tokenizer(settings.ocr_model)
            
            self.ocr_model.eval()
            logger.info("✓ OCR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load OCR model: {e}")
            raise
    
    def _load_reranker(self):
        """Load BAAI reranker model"""
        try:
            logger.info(f"Loading reranker: {settings.reranker_model} on {self.reranker_device}")
            
            # FlagReranker for BAAI reranker
            self.reranker = FlagReranker(
                settings.reranker_model,
                use_fp16=(self.reranker_device == 'cuda')  # Use FP16 on CUDA
            )
            
            # FlagModel handles device internally
            logger.info(f"✓ Reranker loaded successfully on {self.reranker_device}")
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            raise
    
    @torch.inference_mode()
    def encode_text(self, text: str, use_ocr_model: bool = False) -> np.ndarray:
        """
        Generate text embedding using CLIP or OCR model
        
        Args:
            text: Input text
            use_ocr_model: If True, use OCR model; otherwise use CLIP
            
        Returns:
            Normalized embedding vector (512-dim)
        """
        try:
            if use_ocr_model:
                # Use OCR model for multilingual text
                tokens = self.ocr_tokenizer([text]).to(self.device)
                
                with torch.no_grad():
                    text_embed = self.ocr_model.encode_text(tokens)
            else:
                # Use CLIP model for English text
                inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
                
                with torch.no_grad():
                    text_embed = self.clip_model.get_text_features(**inputs)
            
            # Normalize and convert to numpy
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            return text_embed.cpu().float().numpy()[0]
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("CUDA OOM during text encoding. Try reducing batch size or using CPU.")
                torch.cuda.empty_cache()
            raise
    
    def rerank(self, query: str, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Rerank candidates using BAAI reranker
        Args:
            query: Search query
            candidates: List of (text, score) tuples
        Returns:
            Reranked list of (text, score) tuples
        """
        try:
            if not candidates:
                return []
            
            # Prepare input pairs for reranking
            pairs = [[query, text] for text, _ in candidates]
            
            # Get reranking scores
            scores = self.reranker.compute_score(pairs, normalize=True)
            if isinstance(scores, (int, float)):
                scores = [scores]
            
            # Combine with original candidates
            reranked = [(text, float(score)) for (text, _), score in zip(candidates, scores)]
            
            # Sort by new scores (descending)
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            return reranked
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback to original ranking
            return candidates
    
    def clear_cache(self):
        """Clear CUDA cache to free memory"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")


# Global model manager instance
model_manager = ModelManager()