"""Configuration package."""
from .settings import settings
from .aws_config import AWSManager

__all__ = ["settings", "AWSManager"]
