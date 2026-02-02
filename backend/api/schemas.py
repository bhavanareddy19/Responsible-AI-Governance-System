"""Pydantic schemas for API validation."""
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class BaseSchema(BaseModel):
    """Base schema with common config."""
    model_config = ConfigDict(protected_namespaces=())


# === Prediction Schemas ===
class PatientData(BaseSchema):
    """Input schema for patient data."""
    # Vitals
    systolic_bp: float = Field(120, ge=60, le=250, description="Systolic blood pressure")
    diastolic_bp: float = Field(80, ge=40, le=150, description="Diastolic blood pressure")
    heart_rate: float = Field(75, ge=30, le=200, description="Heart rate bpm")
    respiratory_rate: float = Field(16, ge=8, le=40)
    temperature: float = Field(98.6, ge=95, le=105)
    oxygen_saturation: float = Field(97, ge=70, le=100)
    pain_score: float = Field(0, ge=0, le=10)
    
    # Demographics
    age: float = Field(50, ge=0, le=120)
    gender_male: float = Field(0, ge=0, le=1)
    gender_female: float = Field(1, ge=0, le=1) 
    bmi: float = Field(25, ge=10, le=60)
    
    # Key lab values
    hemoglobin: float = Field(13.5, ge=5, le=20)
    creatinine: float = Field(1.0, ge=0.1, le=15)
    glucose: float = Field(100, ge=30, le=600)
    
    # Medical history (binary)
    diabetes: float = Field(0, ge=0, le=1)
    hypertension: float = Field(0, ge=0, le=1)
    heart_disease: float = Field(0, ge=0, le=1)
    previous_admissions_30d: float = Field(0, ge=0, le=10)


class PredictionResponse(BaseSchema):
    """Prediction result with explanation."""
    prediction_id: str
    risk_score: float = Field(ge=0, le=1)
    risk_level: str
    confidence: str
    model_version: str
    timestamp: str
    explanation_available: bool = True


class ExplanationResponse(BaseSchema):
    """SHAP explanation response."""
    prediction_id: str
    risk_score: float
    top_risk_factors: List[Dict[str, Any]]
    protective_factors: List[Dict[str, Any]]
    clinical_rationale: str
    recommendations: List[str]


# === Governance Schemas ===
class BiasReportResponse(BaseSchema):
    """Bias detection report."""
    generated_at: str
    total_samples: int
    overall_bias_detected: bool
    attributes_analyzed: List[str]
    fairness_score: float
    bias_summary: List[Dict[str, Any]]
    recommendations: List[str]


class ComplianceStatusResponse(BaseSchema):
    """Compliance status response."""
    report_id: str
    timestamp: str
    overall_compliant: bool
    compliance_percentage: float
    risk_category: str
    checks_passed: int
    checks_failed: int
    failed_checks: List[Dict[str, Any]]
    all_checks: List[Dict[str, Any]] = []


class AuditEventResponse(BaseSchema):
    """Audit event response."""
    event_id: str
    event_type: str
    timestamp: str
    action: str
    model_version: str
    details: Dict[str, Any]


class AuditQueryRequest(BaseSchema):
    """Audit log query request."""
    event_type: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    limit: int = Field(100, ge=1, le=1000)


# === Monitoring Schemas ===
class HealthResponse(BaseSchema):
    """Health check response."""
    status: str
    version: str
    uptime_hours: float
    model_loaded: bool
    database_connected: bool


class MetricsResponse(BaseSchema):
    """System metrics response."""
    total_predictions: int
    predictions_last_hour: int
    avg_latency_ms: float
    error_rate: float
    high_risk_rate: float
    model_version: str


class PredictionStatsResponse(BaseSchema):
    """Prediction statistics response."""
    window_start: str
    window_end: str
    total_predictions: int
    avg_latency_ms: float
    p95_latency_ms: float
    error_count: int
    high_risk_count: int
    low_risk_count: int
