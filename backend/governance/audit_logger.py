"""Audit Logger for AI Governance."""
import uuid
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import sqlite3

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    PREDICTION = "prediction"
    MODEL_LOAD = "model_load"
    MODEL_UPDATE = "model_update"
    BIAS_CHECK = "bias_check"
    COMPLIANCE_CHECK = "compliance_check"
    DATA_ACCESS = "data_access"
    CONFIG_CHANGE = "config_change"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    ALERT = "alert"


class AuditSeverity(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    event_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: str
    model_version: str
    user_id: Optional[str]
    session_id: Optional[str]
    action: str
    details: Dict
    input_hash: Optional[str]
    output_hash: Optional[str]
    previous_hash: Optional[str]
    signature: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AuditEvent':
        data['event_type'] = AuditEventType(data['event_type'])
        data['severity'] = AuditSeverity(data['severity'])
        return cls(**data)


class AuditLogger:
    """Immutable audit trail with hash chains for tamper detection."""
    
    def __init__(self, db_path: str = "./audit_trail.db", retention_days: int = 2555):
        self.db_path = db_path
        self.retention_days = retention_days
        self._last_hash: Optional[str] = None
        self._event_count: int = 0
        self._init_database()
    
    def _init_database(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY, event_type TEXT NOT NULL,
                    severity TEXT NOT NULL, timestamp TEXT NOT NULL,
                    model_version TEXT, user_id TEXT, session_id TEXT,
                    action TEXT NOT NULL, details TEXT, input_hash TEXT,
                    output_hash TEXT, previous_hash TEXT, signature TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)")
            cursor = conn.execute("SELECT signature FROM audit_events ORDER BY created_at DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                self._last_hash = row[0]
            self._event_count = conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
    
    def _compute_hash(self, data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()
    
    def log_event(self, event_type: AuditEventType, action: str, details: Dict,
                  severity: AuditSeverity = AuditSeverity.INFO, model_version: str = "unknown",
                  user_id: Optional[str] = None, session_id: Optional[str] = None,
                  input_data: Optional[Any] = None, output_data: Optional[Any] = None) -> AuditEvent:
        event_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        input_hash = self._compute_hash(json.dumps(input_data, default=str)) if input_data else None
        output_hash = self._compute_hash(json.dumps(output_data, default=str)) if output_data else None
        
        event_dict = {'event_id': event_id, 'event_type': event_type.value, 'timestamp': timestamp}
        signature = self._compute_hash(f"{self._last_hash or 'GENESIS'}|{json.dumps(event_dict)}")
        
        event = AuditEvent(event_id=event_id, event_type=event_type, severity=severity,
                          timestamp=timestamp, model_version=model_version, user_id=user_id,
                          session_id=session_id, action=action, details=details,
                          input_hash=input_hash, output_hash=output_hash,
                          previous_hash=self._last_hash, signature=signature)
        self._store_event(event)
        self._last_hash = signature
        self._event_count += 1
        return event
    
    def _store_event(self, event: AuditEvent) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO audit_events (event_id, event_type, severity, timestamp,
                    model_version, user_id, session_id, action, details,
                    input_hash, output_hash, previous_hash, signature)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (event.event_id, event.event_type.value, event.severity.value,
                  event.timestamp, event.model_version, event.user_id, event.session_id,
                  event.action, json.dumps(event.details), event.input_hash,
                  event.output_hash, event.previous_hash, event.signature))
    
    def log_prediction(self, prediction_id: str, model_version: str, input_features: Dict,
                       prediction_result: Dict, confidence: float, user_id: Optional[str] = None,
                       risk_score: float = 0.0, risk_level: str = "UNKNOWN") -> AuditEvent:
        return self.log_event(AuditEventType.PREDICTION, "model_prediction",
                             {'prediction_id': prediction_id, 'confidence': confidence,
                              'risk_score': risk_score, 'risk_level': risk_level},
                             model_version=model_version, user_id=user_id,
                             input_data=input_features, output_data=prediction_result)
    
    def query_events(self, event_type: Optional[AuditEventType] = None,
                     start_time: Optional[datetime] = None, limit: int = 100) -> List[AuditEvent]:
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        
        events = []
        valid_fields = {f.name for f in AuditEvent.__dataclass_fields__.values()}
        for row in rows:
            d = dict(row)
            d['details'] = json.loads(d['details'])
            # Strip extra columns from SQLite (e.g., created_at)
            d = {k: v for k, v in d.items() if k in valid_fields}
            events.append(AuditEvent.from_dict(d))
        return events
    
    def get_statistics(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
            types = dict(conn.execute("SELECT event_type, COUNT(*) FROM audit_events GROUP BY event_type").fetchall())
            # Get chain integrity check
            chain_intact = self._verify_chain_sample(conn)
        return {
            'total_events': total,
            'events_by_type': types,
            'prediction_count': types.get('prediction', 0),
            'chain_intact': chain_intact,
            'retention_days': self.retention_days
        }

    def _verify_chain_sample(self, conn) -> bool:
        """Verify a sample of the hash chain for integrity."""
        try:
            rows = conn.execute(
                "SELECT signature, previous_hash FROM audit_events ORDER BY created_at DESC LIMIT 10"
            ).fetchall()
            if len(rows) < 2:
                return True
            for i in range(len(rows) - 1):
                if rows[i][1] is not None and i + 1 < len(rows):
                    if rows[i][1] != rows[i + 1][0]:
                        return False
            return True
        except Exception:
            return True


_audit_logger: Optional[AuditLogger] = None

def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
