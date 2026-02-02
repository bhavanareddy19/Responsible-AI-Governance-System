"""AWS Configuration and Clients."""
import boto3
from botocore.config import Config
from typing import Optional
import logging

from .settings import settings

logger = logging.getLogger(__name__)


class AWSManager:
    """Manages AWS service clients and operations."""
    
    _s3_client = None
    _cloudwatch_client = None
    _cloudwatch_logs_client = None
    
    @classmethod
    def get_boto_config(cls) -> Config:
        """Get boto3 configuration."""
        return Config(
            region_name=settings.AWS_REGION,
            retries={
                'max_attempts': 3,
                'mode': 'standard'
            }
        )
    
    @classmethod
    def get_s3_client(cls):
        """Get or create S3 client."""
        if cls._s3_client is None:
            try:
                cls._s3_client = boto3.client(
                    's3',
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                    config=cls.get_boto_config()
                )
            except Exception as e:
                logger.warning(f"Failed to create S3 client: {e}. Using mock client.")
                cls._s3_client = MockS3Client()
        return cls._s3_client
    
    @classmethod
    def get_cloudwatch_logs_client(cls):
        """Get or create CloudWatch Logs client."""
        if cls._cloudwatch_logs_client is None:
            try:
                cls._cloudwatch_logs_client = boto3.client(
                    'logs',
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                    config=cls.get_boto_config()
                )
            except Exception as e:
                logger.warning(f"Failed to create CloudWatch client: {e}. Using mock client.")
                cls._cloudwatch_logs_client = MockCloudWatchClient()
        return cls._cloudwatch_logs_client
    
    @classmethod
    def upload_model_to_s3(cls, model_path: str, model_key: str) -> bool:
        """Upload a model file to S3."""
        try:
            s3 = cls.get_s3_client()
            s3.upload_file(model_path, settings.S3_BUCKET_NAME, model_key)
            logger.info(f"Model uploaded to s3://{settings.S3_BUCKET_NAME}/{model_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload model to S3: {e}")
            return False
    
    @classmethod
    def download_model_from_s3(cls, model_key: str, local_path: str) -> bool:
        """Download a model file from S3."""
        try:
            s3 = cls.get_s3_client()
            s3.download_file(settings.S3_BUCKET_NAME, model_key, local_path)
            logger.info(f"Model downloaded from s3://{settings.S3_BUCKET_NAME}/{model_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to download model from S3: {e}")
            return False
    
    @classmethod
    def log_to_cloudwatch(cls, log_stream: str, message: str) -> bool:
        """Send log message to CloudWatch."""
        try:
            import time
            logs = cls.get_cloudwatch_logs_client()
            logs.put_log_events(
                logGroupName=settings.CLOUDWATCH_LOG_GROUP,
                logStreamName=log_stream,
                logEvents=[{
                    'timestamp': int(time.time() * 1000),
                    'message': message
                }]
            )
            return True
        except Exception as e:
            logger.error(f"Failed to log to CloudWatch: {e}")
            return False


class MockS3Client:
    """Mock S3 client for local development."""
    
    def upload_file(self, *args, **kwargs):
        logger.info("MockS3: upload_file called")
    
    def download_file(self, *args, **kwargs):
        logger.info("MockS3: download_file called")
    
    def list_objects_v2(self, *args, **kwargs):
        return {'Contents': []}


class MockCloudWatchClient:
    """Mock CloudWatch client for local development."""
    
    def put_log_events(self, *args, **kwargs):
        logger.info("MockCloudWatch: put_log_events called")
    
    def create_log_stream(self, *args, **kwargs):
        logger.info("MockCloudWatch: create_log_stream called")
