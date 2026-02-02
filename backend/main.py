"""FastAPI Main Application - Responsible AI Governance System."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from config import settings
from api import predictions_router, governance_router, monitoring_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Initialize model on startup
    from ml.models import create_default_model
    model = create_default_model()
    logger.info(f"Model loaded: {model.model_version} ({sum(p.numel() for p in model.parameters())} parameters)")

    # Initialize audit logger
    from governance import get_audit_logger, AuditEventType
    audit = get_audit_logger()
    logger.info("Audit logger initialized")

    # Log system startup event
    audit.log_event(
        event_type=AuditEventType.SYSTEM_EVENT,
        action="system_startup",
        model_version=model.model_version,
        details={
            "app_name": settings.APP_NAME,
            "app_version": settings.APP_VERSION,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "architecture": "HealthcareRiskModel [256, 128, 64]"
        }
    )

    # Seed initial predictions so dashboard has data on first load
    stats = audit.get_statistics()
    if stats['total_events'] < 10:
        logger.info("Seeding initial demo data...")
        try:
            from seed_data import seed_synthetic_data
            seed_synthetic_data(100)
            logger.info("Initial data seeded successfully")
        except Exception as e:
            logger.warning(f"Failed to seed initial data: {e}")

    yield

    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="""
    ## Responsible AI Governance System
    
    A comprehensive AI governance framework for healthcare applications featuring:
    
    - **Model Transparency**: Full visibility into model predictions and behavior
    - **Bias Detection**: Automated detection of demographic bias and fairness metrics
    - **Explainability**: SHAP-based explanations with clinical rationale
    - **Audit Trails**: Immutable logging for regulatory compliance
    - **Real-time Monitoring**: Track predictions, latency, and system health
    
    ### Healthcare Compliance
    - HIPAA compliant data handling
    - FDA AI/ML guidance alignment
    - 7-year audit log retention
    """,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predictions_router, prefix=settings.API_PREFIX)
app.include_router(governance_router, prefix=settings.API_PREFIX)
app.include_router(monitoring_router, prefix=settings.API_PREFIX)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "api_prefix": settings.API_PREFIX
    }


@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "version": "v1",
        "endpoints": {
            "predictions": f"{settings.API_PREFIX}/predictions",
            "governance": f"{settings.API_PREFIX}/governance",
            "monitoring": f"{settings.API_PREFIX}/monitoring"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
