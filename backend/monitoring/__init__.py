"""Monitoring module for AI Governance System."""
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class PredictionMetrics:
    """Metrics for a time window."""
    window_start: str
    window_end: str
    total_predictions: int
    avg_latency_ms: float
    p95_latency_ms: float
    error_count: int
    high_risk_count: int
    low_risk_count: int


class PredictionMonitor:
    """Real-time prediction monitoring for governance."""
    
    def __init__(self, window_size_seconds: int = 60):
        self.window_size = window_size_seconds
        self._predictions: List[Dict] = []
        self._lock = threading.Lock()
        self.start_time = datetime.utcnow()
    
    def record_prediction(self, prediction_id: str, latency_ms: float, 
                         risk_score: float, success: bool = True) -> None:
        """Record a prediction event."""
        with self._lock:
            self._predictions.append({
                'timestamp': datetime.utcnow(), 'prediction_id': prediction_id,
                'latency_ms': latency_ms, 'risk_score': risk_score, 'success': success
            })
            # Keep only recent predictions
            cutoff = datetime.utcnow() - timedelta(hours=1)
            self._predictions = [p for p in self._predictions if p['timestamp'] > cutoff]
    
    def get_current_metrics(self) -> PredictionMetrics:
        """Get metrics for current window."""
        with self._lock:
            now = datetime.utcnow()
            window_start = now - timedelta(seconds=self.window_size)
            window_preds = [p for p in self._predictions if p['timestamp'] > window_start]
            
            if not window_preds:
                return PredictionMetrics(
                    window_start=window_start.isoformat(), window_end=now.isoformat(),
                    total_predictions=0, avg_latency_ms=0, p95_latency_ms=0,
                    error_count=0, high_risk_count=0, low_risk_count=0
                )
            
            latencies = [p['latency_ms'] for p in window_preds]
            latencies.sort()
            
            return PredictionMetrics(
                window_start=window_start.isoformat(), window_end=now.isoformat(),
                total_predictions=len(window_preds),
                avg_latency_ms=sum(latencies) / len(latencies),
                p95_latency_ms=latencies[int(len(latencies) * 0.95)] if latencies else 0,
                error_count=sum(1 for p in window_preds if not p['success']),
                high_risk_count=sum(1 for p in window_preds if p['risk_score'] >= 0.7),
                low_risk_count=sum(1 for p in window_preds if p['risk_score'] < 0.3)
            )
    
    def get_statistics(self) -> Dict:
        """Get overall statistics."""
        with self._lock:
            total = len(self._predictions)
            if total == 0:
                return {'total_predictions': 0, 'uptime_hours': 0}
            
            latencies = [p['latency_ms'] for p in self._predictions]
            uptime = (datetime.utcnow() - self.start_time).total_seconds() / 3600
            
            return {
                'total_predictions': total,
                'uptime_hours': round(uptime, 2),
                'avg_latency_ms': round(sum(latencies) / len(latencies), 2),
                'error_rate': sum(1 for p in self._predictions if not p['success']) / total,
                'high_risk_rate': sum(1 for p in self._predictions if p['risk_score'] >= 0.7) / total
            }


class GovernanceLogger:
    """Structured logging for governance events."""
    
    def __init__(self):
        self.events: List[Dict] = []
        self._lock = threading.Lock()
    
    def log(self, event_type: str, message: str, details: Optional[Dict] = None) -> None:
        """Log a governance event."""
        with self._lock:
            self.events.append({
                'timestamp': datetime.utcnow().isoformat(),
                'type': event_type,
                'message': message,
                'details': details or {}
            })
            # Keep last 10000 events
            if len(self.events) > 10000:
                self.events = self.events[-5000:]
    
    def get_recent_events(self, limit: int = 100, event_type: Optional[str] = None) -> List[Dict]:
        """Get recent events."""
        with self._lock:
            events = self.events if not event_type else [e for e in self.events if e['type'] == event_type]
            return events[-limit:][::-1]


# Global instances
_prediction_monitor: Optional[PredictionMonitor] = None
_governance_logger: Optional[GovernanceLogger] = None


def get_prediction_monitor() -> PredictionMonitor:
    global _prediction_monitor
    if _prediction_monitor is None:
        _prediction_monitor = PredictionMonitor()
    return _prediction_monitor


def get_governance_logger() -> GovernanceLogger:
    global _governance_logger
    if _governance_logger is None:
        _governance_logger = GovernanceLogger()
    return _governance_logger
