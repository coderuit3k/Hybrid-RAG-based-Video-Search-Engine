from pydantic_settings import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    """Application configuration settings"""
    
    # Milvus settings
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "video_search"
    
    # Data directories (relative to backend/ folder)
    data_dir: str = "../data"
    keyframes_dir: str = "../data/keyframes"
    embed_kf_dir: str = "../data/embed_kf"
    embed_ocr_dir: str = "../data/embed_ocr"
    embed_obj_dir: str = "../data/embed_od"
    merged_dir: str = "../data/MERGED"
    
    # Model settings
    model_device: str = "cuda"
    reranker_device: str = "cuda"
    translation_device: str = "cuda"
    
    # Model names
    clip_model: str = "openai/clip-vit-base-patch32"
    ocr_model: str = "xlm-roberta-base-ViT-B-32"
    ocr_pretrained: str = "laion5b_s13b_b90k"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    translation_model: str = "Helsinki-NLP/opus-mt-vi-en"
    
    # Search parameters
    embedding_dim: int = 512
    top_k: int = 100  # Initial candidates from Milvus
    rerank_top_k: int = 50  # How many to rerank
    final_top_k: int = 30  # Final results to return
    
    # Hybrid search weights
    kf_weight: float = 0.5
    ocr_weight: float = 0.3
    obj_weight: float = 0.2
    
    # API settings
    api_title: str = "Video Search API"
    api_version: str = "1.0.0"
    cors_origins: List[str] = ["*"]
    
    # AWS S3 Configuration
    use_s3: bool = True
    s3_bucket_name: str = ""
    s3_region: str = "ap-southeast-2"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()