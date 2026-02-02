"""Application configuration settings."""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    APP_NAME: str = "Responsible AI Governance System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./governance.db"
    
    # AWS Configuration
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = "ai-governance-models"
    CLOUDWATCH_LOG_GROUP: str = "/ai-governance/predictions"
    
    # Model Configuration
    MODEL_PATH: str = "./models/healthcare_model.pt"
    MODEL_VERSION: str = "1.0.0"
    
    # Governance
    AUDIT_LOG_RETENTION_DAYS: int = 2555  # 7 years for healthcare compliance
    BIAS_THRESHOLD: float = 0.1  # Maximum acceptable bias score
    
    # Monitoring
    PREDICTION_BATCH_SIZE: int = 1000
    METRICS_INTERVAL_SECONDS: int = 60
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
