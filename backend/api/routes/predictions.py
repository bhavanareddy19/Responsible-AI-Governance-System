"""Prediction API routes."""
import uuid
import time
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException
from typing import Dict

from api.schemas import PatientData, PredictionResponse, ExplanationResponse
from ml.models import ALL_FEATURES

router = APIRouter(prefix="/predictions", tags=["Predictions"])

# Global state (in production, use dependency injection)
_model = None
_explainer = None
_audit_logger = None
_monitor = None
_prediction_cache: Dict[str, Dict] = {}  # Store recent predictions for explain endpoint


def get_model():
    global _model
    if _model is None:
        from ml.models import create_default_model
        _model = create_default_model()
        _model.eval()
    return _model


def get_explainer():
    global _explainer
    if _explainer is None:
        from explainability import SHAPExplainer
        model = get_model()
        _explainer = SHAPExplainer(model, ALL_FEATURES)
    return _explainer


def get_audit_logger():
    global _audit_logger
    if _audit_logger is None:
        from governance import get_audit_logger as get_logger
        _audit_logger = get_logger()
    return _audit_logger


def get_monitor():
    global _monitor
    if _monitor is None:
        from monitoring import get_prediction_monitor
        _monitor = get_prediction_monitor()
    return _monitor


def patient_to_features(patient: PatientData) -> np.ndarray:
    """Convert patient data to feature array."""
    features = np.zeros(len(ALL_FEATURES))
    patient_dict = patient.model_dump()
    
    feature_idx = {name: i for i, name in enumerate(ALL_FEATURES)}
    for key, value in patient_dict.items():
        if key in feature_idx:
            features[feature_idx[key]] = value
    
    return features.reshape(1, -1)


@router.post("", response_model=PredictionResponse)
async def make_prediction(patient: PatientData) -> PredictionResponse:
    """Make a risk prediction for a patient."""
    import torch
    from datetime import datetime
    
    start_time = time.time()
    prediction_id = str(uuid.uuid4())
    
    try:
        model = get_model()
        features = patient_to_features(patient)
        
        with torch.no_grad():
            tensor = torch.FloatTensor(features)
            prediction = model(tensor).item()
        
        risk_level = "HIGH" if prediction >= 0.7 else "MODERATE" if prediction >= 0.4 else "LOW"
        confidence = "High" if abs(prediction - 0.5) > 0.3 else "Moderate"
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Record monitoring
        monitor = get_monitor()
        monitor.record_prediction(prediction_id, latency_ms, prediction, True)
        
        # Log to audit trail
        audit = get_audit_logger()
        audit.log_prediction(
            prediction_id=prediction_id,
            model_version=model.model_version,
            input_features=patient.model_dump(),
            prediction_result={'risk_score': prediction, 'risk_level': risk_level},
            confidence=prediction,
            risk_score=prediction,
            risk_level=risk_level
        )

        # Cache prediction data for explain endpoint
        _prediction_cache[prediction_id] = {
            'patient_data': patient.model_dump(),
            'risk_score': prediction,
            'risk_level': risk_level
        }
        # Keep cache bounded
        if len(_prediction_cache) > 1000:
            oldest_keys = list(_prediction_cache.keys())[:500]
            for k in oldest_keys:
                del _prediction_cache[k]

        return PredictionResponse(
            prediction_id=prediction_id,
            risk_score=round(prediction, 4),
            risk_level=risk_level,
            confidence=confidence,
            model_version=model.model_version,
            timestamp=datetime.utcnow().isoformat(),
            explanation_available=True
        )
    
    except Exception as e:
        monitor = get_monitor()
        monitor.record_prediction(prediction_id, 0, 0, False)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{prediction_id}/explain", response_model=ExplanationResponse)
async def get_explanation(prediction_id: str) -> ExplanationResponse:
    """Get SHAP explanation for a prediction."""
    from explainability import ClinicalRationaleGenerator

    # Retrieve stored prediction data from cache
    cached = _prediction_cache.get(prediction_id)
    if cached:
        patient = PatientData(**cached['patient_data'])
    else:
        patient = PatientData()  # Fallback to defaults

    explainer = get_explainer()
    features = patient_to_features(patient)

    explanation = explainer.explain_prediction(features, prediction_id)
    rationale = ClinicalRationaleGenerator.generate_clinical_rationale(explanation)

    return ExplanationResponse(
        prediction_id=prediction_id,
        risk_score=explanation.prediction_value,
        top_risk_factors=explanation.top_positive_features,
        protective_factors=explanation.top_negative_features,
        clinical_rationale=explanation.explanation_text,
        recommendations=rationale.get('clinical_recommendations', [])
    )


@router.post("/batch")
async def batch_predict(patients: list[PatientData]) -> Dict:
    """Batch prediction for multiple patients."""
    import torch
    from datetime import datetime
    
    model = get_model()
    results = []
    
    for patient in patients:
        features = patient_to_features(patient)
        with torch.no_grad():
            tensor = torch.FloatTensor(features)
            prediction = model(tensor).item()
        
        results.append({
            'prediction_id': str(uuid.uuid4()),
            'risk_score': round(prediction, 4),
            'risk_level': "HIGH" if prediction >= 0.7 else "MODERATE" if prediction >= 0.4 else "LOW"
        })
    
    return {
        'batch_id': str(uuid.uuid4()),
        'count': len(results),
        'timestamp': datetime.utcnow().isoformat(),
        'predictions': results
    }
