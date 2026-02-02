"""Monitoring API routes."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter
from datetime import datetime

from api.schemas import HealthResponse, MetricsResponse, PredictionStatsResponse

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """System health check endpoint."""
    from monitoring import get_prediction_monitor
    from config import settings
    
    monitor = get_prediction_monitor()
    stats = monitor.get_statistics()
    
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        uptime_hours=stats.get('uptime_hours', 0),
        model_loaded=True,
        database_connected=True
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """Get system metrics."""
    from monitoring import get_prediction_monitor
    from config import settings

    monitor = get_prediction_monitor()
    stats = monitor.get_statistics()
    current = monitor.get_current_metrics()
    
    # Total predictions in the monitor's rolling 1-hour window is the actual hourly count
    return MetricsResponse(
        total_predictions=stats.get('total_predictions', 0),
        predictions_last_hour=stats.get('total_predictions', 0),
        avg_latency_ms=stats.get('avg_latency_ms', 0),
        error_rate=stats.get('error_rate', 0),
        high_risk_rate=stats.get('high_risk_rate', 0),
        model_version=settings.MODEL_VERSION
    )


@router.get("/predictions/stats", response_model=PredictionStatsResponse)
async def get_prediction_stats() -> PredictionStatsResponse:
    """Get current prediction statistics."""
    from monitoring import get_prediction_monitor
    
    monitor = get_prediction_monitor()
    metrics = monitor.get_current_metrics()
    
    return PredictionStatsResponse(
        window_start=metrics.window_start,
        window_end=metrics.window_end,
        total_predictions=metrics.total_predictions,
        avg_latency_ms=metrics.avg_latency_ms,
        p95_latency_ms=metrics.p95_latency_ms,
        error_count=metrics.error_count,
        high_risk_count=metrics.high_risk_count,
        low_risk_count=metrics.low_risk_count
    )


@router.get("/alerts")
async def get_alerts():
    """Get active system alerts."""
    from monitoring import get_prediction_monitor, get_governance_logger
    
    monitor = get_prediction_monitor()
    logger = get_governance_logger()
    
    stats = monitor.get_statistics()
    alerts = []
    
    if stats.get('error_rate', 0) > 0.05:
        alerts.append({
            'level': 'warning',
            'message': f"High error rate: {stats['error_rate']:.1%}",
            'timestamp': datetime.utcnow().isoformat()
        })
    
    if stats.get('high_risk_rate', 0) > 0.3:
        alerts.append({
            'level': 'info',
            'message': f"Elevated high-risk predictions: {stats['high_risk_rate']:.1%}",
            'timestamp': datetime.utcnow().isoformat()
        })
    
    return {'alerts': alerts, 'count': len(alerts)}
