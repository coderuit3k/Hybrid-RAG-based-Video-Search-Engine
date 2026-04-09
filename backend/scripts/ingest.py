import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.database.milvus_client import milvus_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_embedding(video_id: str, frame_idx: int, embed_dir: str) -> np.ndarray:
    """
    Load embedding from .npy file
    
    Args:
        video_id: Video ID
        frame_idx: Frame index
        embed_dir: Directory containing embeddings
        
    Returns:
        Embedding vector as numpy array
    """
    filename = f"{video_id}_{frame_idx}.npy"
    filepath = os.path.join(embed_dir, filename)
    
    if not os.path.exists(filepath):
        logger.warning(f"Embedding file not found: {filepath}")
        return np.zeros(settings.embedding_dim, dtype=np.float32)
    
    try:
        embedding = np.load(filepath)
        
        # Ensure correct shape
        if embedding.shape != (settings.embedding_dim,):
            logger.warning(f"Unexpected embedding shape {embedding.shape} for {filename}")
            return np.zeros(settings.embedding_dim, dtype=np.float32)
        
        return embedding.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return np.zeros(settings.embedding_dim, dtype=np.float32)


def read_csv_metadata(merged_dir: str) -> pd.DataFrame:
    """
    Read all CSV files from MERGED directory
    
    Args:
        merged_dir: Directory containing CSV files
        
    Returns:
        Combined DataFrame
    """
    csv_files = sorted(list(Path(merged_dir).glob("*.csv")))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {merged_dir}")
    
    logger.info(f"Found {len(csv_files)} CSV files")
    
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to read {csv_file}: {e}")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined_df)} rows from CSV files")
    
    return combined_df


def prepare_batch_data(df_batch: pd.DataFrame) -> Dict[str, List]:
    """
    Prepare batch data for Milvus insertion
    
    Args:
        df_batch: DataFrame batch
        
    Returns:
        Dictionary with field names and values
    """
    batch_data = {
        "video_id": [],
        "frame_idx": [],
        "image_path": [],
        "ocr_text": [],
        "kf_embedding": [],
        "ocr_embedding": [],
        "obj_embedding": []
    }
    
    for _, row in df_batch.iterrows():
        video_id = str(row['video_id'])
        frame_idx = int(row['frame_idx'])
        image_path = str(row.get('image_path', ''))
        ocr_text = str(row.get('ocr_text', ''))
        
        # Load embeddings
        kf_emb = load_embedding(video_id, frame_idx, settings.embed_kf_dir)
        ocr_emb = load_embedding(video_id, frame_idx, settings.embed_ocr_dir)
        obj_emb = load_embedding(video_id, frame_idx, settings.embed_obj_dir)
        
        batch_data["video_id"].append(video_id)
        batch_data["frame_idx"].append(frame_idx)
        batch_data["image_path"].append(image_path)
        batch_data["ocr_text"].append(ocr_text)
        batch_data["kf_embedding"].append(kf_emb.tolist())
        batch_data["ocr_embedding"].append(ocr_emb.tolist())
        batch_data["obj_embedding"].append(obj_emb.tolist())
        
    # Debug: print structure of first item
    if len(batch_data["video_id"]) > 0:
        logger.debug(f"Sample data types: video_id={type(batch_data['video_id'][0])}, "
                    f"frame_idx={type(batch_data['frame_idx'][0])}, "
                    f"kf_embedding length={len(batch_data['kf_embedding'][0])}")
    
    return batch_data


def ingest_data(batch_size: int = 500, recreate_collection: bool = False):
    """
    Main ingestion function
    
    Args:
        batch_size: Number of rows to insert per batch
        recreate_collection: Whether to recreate the collection
    """
    try:
        # Connect to Milvus
        logger.info("Connecting to Milvus...")
        milvus_client.connect()
        
        # Create or load collection
        if recreate_collection:
            logger.info("Creating new collection...")
            milvus_client.create_collection(drop_existing=True)
        else:
            if not milvus_client.collection:
                logger.info("Collection not found, creating new one...")
                milvus_client.create_collection(drop_existing=False)
        
        collection = milvus_client.get_collection()
        
        # Read CSV metadata
        logger.info(f"Reading CSV files from {settings.merged_dir}...")
        df = read_csv_metadata(settings.merged_dir)
        
        # Batch insert
        total_rows = len(df)
        num_batches = (total_rows + batch_size - 1) // batch_size
        
        logger.info(f"Starting ingestion: {total_rows} rows in {num_batches} batches")
        
        inserted_count = 0
        
        for i in tqdm(range(0, total_rows, batch_size), desc="Ingesting batches"):
            batch_df = df.iloc[i: i + batch_size]
            
            # Prepare batch data
            batch_data = prepare_batch_data(batch_df)
            
            # Convert dict of lists to list of dicts for Milvus insert
            entities = []
            num_entities = len(batch_data["video_id"])
            for j in range(num_entities):
                entity = {
                    "video_id": batch_data["video_id"][j],
                    "frame_idx": batch_data["frame_idx"][j],
                    "image_path": batch_data["image_path"][j],
                    "ocr_text": batch_data["ocr_text"][j],
                    "kf_embedding": batch_data["kf_embedding"][j],
                    "ocr_embedding": batch_data["ocr_embedding"][j],
                    "obj_embedding": batch_data["obj_embedding"][j]
                }
                entities.append(entity)
            
            # Insert into Milvus
            try:
                # Add timeout to insert to prevent hanging
                result = collection.insert(entities, timeout=60)
                inserted_count += num_entities
                
                # Flush periodically
                if (i // batch_size + 1) % 10 == 0:
                    collection.flush()
                    # Small sleep to let Milvus catch up
                    import time
                    time.sleep(0.2)
                    
            except Exception as e:
                logger.error(f"Failed to insert batch {i // batch_size}: {e}")
                continue
        
        # Final flush
        logger.info("Flushing final data...")
        collection.flush()
        
        # Verify count
        actual_count = collection.num_entities
        logger.info(f"Ingestion complete!")
        logger.info(f"  - Expected: {total_rows}")
        logger.info(f"  - Inserted: {inserted_count}")
        logger.info(f"  - In collection: {actual_count}")
        
        # Load collection for searching
        collection.load()
        logger.info("Collection loaded and ready for search")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise
    finally:
        milvus_client.disconnect()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Ingest embeddings into Milvus")
    parser.add_argument("--batch-size", type=int, default=500, help="Number of rows to insert per batch (default: 500)")
    parser.add_argument("--recreate-collection", action="store_true", help="Drop and recreate the collection")
    
    args = parser.parse_args()
    logger.info("Milvus Data Ingestion Script")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Recreate collection: {args.recreate_collection}")
    logger.info(f"Data directory: {settings.data_dir}")
    
    # Insert data into Milvus
    ingest_data(batch_size=args.batch_size, recreate_collection=args.recreate_collection)


if __name__ == "__main__":
    main()
