import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import logging
from typing import Optional, Union
from pathlib import Path
from dotenv import load_dotenv
from app.config import settings

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class S3Service:
    """Service for AWS S3 operations"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.enabled = settings.use_s3
        self.bucket = os.getenv('S3_BUCKET_NAME')
        self.region = os.getenv('S3_REGION')
        self.s3_client = None
        
        if self.enabled:
            logger.info(f"Initializing S3Service for bucket: {self.bucket}")
            try:
                # Initialize boto3 client
                kwargs = {
                    'region_name': self.region
                }
                
                access_key = os.getenv('S3_ACCESS_KEY')
                secret_key = os.getenv('S3_SECRET_KEY')
                
                if access_key and secret_key:
                    kwargs['aws_access_key_id'] = access_key
                    kwargs['aws_secret_access_key'] = secret_key
                
                self.s3_client = boto3.client('s3', **kwargs)
                logger.info("S3 Client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {e}")
                self.enabled = False
        else:
            logger.info("S3Service is disabled")
            
        self._initialized = True
    
    def upload_file(self, file_path: Union[str, Path], object_name: Optional[str] = None) -> Optional[str]:
        """
        Upload a file to S3
        
        Args:
            file_path: Path to local file
            object_name: S3 object key (defaults to file name)
            
        Returns:
            Public S3 URL or None if failed
        """
        if not self.enabled or not self.s3_client:
            logger.warning("S3 is disabled, skipping upload")
            return None
            
        try:
            file_path = str(file_path)
            if object_name is None:
                object_name = os.path.basename(file_path)
            
            # Use forward slashes for S3 keys even on Windows
            object_name = object_name.replace(os.path.sep, '/')
            
            logger.info(f"Uploading {file_path} to s3://{self.bucket}/{object_name}")
            
            # Upload file
            self.s3_client.upload_file(
                file_path, 
                self.bucket, 
                object_name,
                ExtraArgs={'ContentType': self._guess_content_type(file_path)}
            )
            
            # Return public URL (assuming bucket policy allows public read or we use presigned)
            # Standard S3 URL format
            url = f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{object_name}"
            return url
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return None
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            return None
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return None
    
    def check_file_exists(self, object_name: str) -> bool:
        """Check if file exists in S3"""
        if not self.enabled or not self.s3_client:
            return False
            
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=object_name)
            return True
        except ClientError:
            return False
            
    def _guess_content_type(self, file_path: str) -> str:
        """Guess content type based on extension"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            return 'image/jpeg'
        elif ext == '.png':
            return 'image/png'
        elif ext == '.mp4':
            return 'video/mp4'
        elif ext == '.json':
            return 'application/json'
        elif ext == '.csv':
            return 'text/csv'
        return 'application/octet-stream'


s3_service = S3Service()