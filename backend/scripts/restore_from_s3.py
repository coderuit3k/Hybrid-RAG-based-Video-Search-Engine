import os
import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import boto3

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def restore_data(dry_run: bool = False):
    """
    Restore backup data from S3
    """
    if not settings.use_s3:
        logger.error("S3 is disabled in config. Please set use_s3=True.")
        return

    # Direct boto3 usage to list objects
    s3_client = boto3.client(
        's3',
        region_name=settings.s3_region,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key
    )
    
    bucket = settings.s3_bucket_name
    data_dir = Path(settings.data_dir)
    
    logger.info(f"Starting restore from bucket: {bucket}")
    
    # Folders to restore
    folders = ["MERGED", "embeddings"]
    
    for folder in folders:
        prefix = f"{folder}/"
        logger.info(f"Listing files in {prefix}...")
        
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        files_to_download = []
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    files_to_download.append(obj['Key'])
                    
        logger.info(f"Found {len(files_to_download)} files in {prefix}")
        
        if not dry_run:
            for s3_key in tqdm(files_to_download, desc=f"Restoring {folder}"):
                target_path = data_dir / s3_key
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    s3_client.download_file(bucket, s3_key, str(target_path))
                except Exception as e:
                    logger.error(f"Failed to download {s3_key}: {e}")
        else:
            logger.info("Dry run: Skipping download")
            
    logger.info("Restore complete!")
    logger.info("You can now run 'python backend/scripts/ingest.py --recreate-collection' to load this data into Milvus.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restore backup from S3")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without changes")
    args = parser.parse_args()
    
    restore_data(dry_run=args.dry_run)