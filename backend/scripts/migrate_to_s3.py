import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.services.s3_service import s3_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_data(dry_run: bool = False):
    """
    Migrate data to S3
    
    Args:
        dry_run: If True, don't actually upload or modify files
    """
    if not settings.use_s3:
        logger.error("S3 is disabled in config. Please set use_s3=True.")
        return

    data_dir = Path(settings.data_dir)
    keyframes_dir = data_dir / "keyframes"
    merged_dir = data_dir / "MERGED"
        
    logger.info(f"Starting migration to bucket: {settings.s3_bucket_name}")
    
    # 1. Upload Keyframes
    # Find all images recursively
    files_to_upload = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        files_to_upload.extend(list(keyframes_dir.rglob(ext)))
    
    logger.info(f"Found {len(files_to_upload)} images to upload")
    
    if not dry_run:
        for file_path in tqdm(files_to_upload, desc="Uploading Keyframes"):
            # Construct S3 key: keyframes/Lxx/Vxx/filename.jpg
            # Relative path from data_dir
            rel_path = file_path.relative_to(data_dir)
            s3_key = str(rel_path).replace(os.path.sep, '/')
            
            # Upload
            s3_service.upload_file(file_path, s3_key)
    else:
        logger.info("Dry run: Skipping keyframe upload")

    # 2. Update CSV Metadata
    logger.info("Updating CSV metadata...")
    csv_files = list(merged_dir.glob("*.csv"))
    
    for csv_file in csv_files:
        logger.info(f"Processing {csv_file.name}")
        if dry_run:
            continue
            
        try:
            df = pd.read_csv(csv_file)
            
            # Function to update path
            def update_path(path):
                # If it's already a URL, skip
                if str(path).startswith(('http', 's3://')):
                    return path
                
                # Normalize path separators
                path_str = str(path).replace('\\', '/')
                
                if 'keyframes' in path_str:
                    idx = path_str.find('keyframes')
                    if idx != -1:
                        s3_key = path_str[idx:]
                        return f"https://{settings.s3_bucket_name}.s3.{settings.s3_region}.amazonaws.com/{s3_key}"
                
                return path

            # Apply update
            df['image_path'] = df['image_path'].apply(update_path)
            
            # Save back
            df.to_csv(csv_file, index=False)
            logger.info(f"Updated {csv_file.name}")
            
        except Exception as e:
            logger.error(f"Failed to process {csv_file}: {e}")

    # 3. Upload Backup (MERGED + Embeddings)
    logger.info("Uploading Backups (Metadata & Embeddings)...")
    
    dirs_to_backup = [
        data_dir / "MERGED",
        data_dir / "embeddings"
    ]
    
    if not dry_run:
        for backup_dir in dirs_to_backup:
            if not backup_dir.exists():
                logger.warning(f"Backup dir not found: {backup_dir}")
                continue
                
            backup_files = list(backup_dir.rglob("*"))
            backup_files = [f for f in backup_files if f.is_file()]
            
            for file_path in tqdm(backup_files, desc=f"Backing up {backup_dir.name}"):
                rel_path = file_path.relative_to(data_dir) # e.g. MERGED/L01.csv
                s3_key = str(rel_path).replace(os.path.sep, '/')
                
                s3_service.upload_file(file_path, s3_key)
    else:
        logger.info("Dry run: Skipping backup upload")

    logger.info("Migration complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate data to S3")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without changes")
    args = parser.parse_args()
    
    migrate_data(dry_run=args.dry_run)
