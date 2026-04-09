import logging
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from typing import Optional
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MilvusClient:
    """Client for Milvus vector database operations"""
    
    def __init__(self):
        self.collection: Optional[Collection] = None
        self.collection_name = settings.milvus_collection
    
    def connect(self):
        """Establish connection to Milvus"""
        try:
            logger.info(f"Connecting to Milvus at {settings.milvus_host}:{settings.milvus_port}")
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port
            )
            logger.info("Connected to Milvus")
            
            # Load or create collection
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.collection.load()
                logger.info(f"Loaded existing collection: {self.collection_name}")
            else:
                logger.warning(f"Collection {self.collection_name} does not exist. Create it using ingest.py")
                
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from Milvus"""
        try:
            connections.disconnect(alias="default")
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {e}")
    
    def create_collection(self, drop_existing: bool = False):
        """
        Create collection with schema for multimodal embeddings
        Args:
            drop_existing: Drop collection if it already exists
        """
        try:
            # Drop existing collection if requested
            if drop_existing and utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"Dropped existing collection: {self.collection_name}")
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=16),
                FieldSchema(name="frame_idx", dtype=DataType.INT64),
                FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="ocr_text", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="kf_embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.embedding_dim),
                FieldSchema(name="ocr_embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.embedding_dim),
                FieldSchema(name="obj_embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.embedding_dim),
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="Multimodal video keyframe embeddings"
            )
            
            # Create collection
            self.collection = Collection(
                name=self.collection_name,
                schema=schema,
                using='default'
            )
            
            logger.info(f"Created collection: {self.collection_name}")
            
            # Create indexes for each vector field
            index_params = {
                "metric_type": "IP",  # Inner Product (cosine similarity for normalized vectors)
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            
            for field_name in ["kf_embedding", "ocr_embedding", "obj_embedding"]:
                self.collection.create_index(
                    field_name=field_name,
                    index_params=index_params
                )
                logger.info(f"✓ Created index on {field_name}")
            
            return self.collection
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def get_collection(self) -> Collection:
        """Get the current collection"""
        if self.collection is None:
            raise ValueError("Collection not initialized. Call connect() first.")
        return self.collection
    
    def get_stats(self) -> dict:
        """Get collection statistics"""
        try:
            if self.collection is None:
                return {"error": "Collection not loaded"}
            
            stats = {
                "name": self.collection_name,
                "num_entities": self.collection.num_entities,
                "is_loaded": utility.load_state(self.collection_name),
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

milvus_client = MilvusClient()