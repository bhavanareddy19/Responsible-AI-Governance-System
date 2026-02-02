"""API routes package."""
from .predictions import router as predictions_router
from .governance import router as governance_router
from .monitoring import router as monitoring_router

__all__ = ["predictions_router", "governance_router", "monitoring_router"]
