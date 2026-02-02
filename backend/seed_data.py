"""Data seeding script - generates synthetic predictions and audit logs."""
import sys
import os
import random
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.models import create_default_model, ALL_FEATURES
from ml.training import DataGenerator
from governance import get_audit_logger, AuditEventType
from monitoring import get_prediction_monitor
import torch
import numpy as np


def seed_synthetic_data(num_predictions: int = 500):
    """Seed the system with synthetic prediction data."""
    print(f"[SEED] Seeding {num_predictions} synthetic predictions...")
    
    model = create_default_model()
    model.eval()
    
    monitor = get_prediction_monitor()
    audit = get_audit_logger()
    
    # Generate synthetic patient data
    X, y, df = DataGenerator.generate_synthetic_data(n_samples=num_predictions)
    
    high_risk = 0
    low_risk = 0
    moderate_risk = 0
    
    for i in range(num_predictions):
        prediction_id = f"seed-{i:05d}"
        
        # Make prediction
        with torch.no_grad():
            features = torch.FloatTensor(X[i:i+1])
            risk_score = model(features).item()
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = "HIGH"
            high_risk += 1
        elif risk_score >= 0.4:
            risk_level = "MODERATE"
            moderate_risk += 1
        else:
            risk_level = "LOW"
            low_risk += 1
        
        # Simulate realistic latency
        latency_ms = random.uniform(15, 45)
        
        # Record to monitor
        monitor.record_prediction(prediction_id, latency_ms, risk_score, True)
        
        # Log to audit
        audit.log_prediction(
            prediction_id=prediction_id,
            model_version=model.model_version,
            input_features={name: float(X[i][j]) for j, name in enumerate(ALL_FEATURES)},
            prediction_result={'risk_score': risk_score, 'risk_level': risk_level},
            confidence=abs(risk_score - 0.5) * 2,
            risk_score=risk_score,
            risk_level=risk_level
        )
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{num_predictions} predictions...")
    
    # Log some governance events
    audit.log_event(
        event_type=AuditEventType.BIAS_CHECK,
        action="scheduled_bias_analysis",
        model_version=model.model_version,
        details={"passed": True, "attributes": ["gender", "age_group"], "fairness_score": 94.2}
    )
    
    audit.log_event(
        event_type=AuditEventType.COMPLIANCE_CHECK,
        action="hipaa_fda_compliance_verification",
        model_version=model.model_version,
        details={"passed": True, "checks": 7, "compliance_percentage": 100.0}
    )
    
    audit.log_event(
        event_type=AuditEventType.MODEL_LOAD,
        action="model_initialization",
        model_version=model.model_version,
        details={"parameters": sum(p.numel() for p in model.parameters()), "architecture": "HealthcareRiskModel"}
    )
    
    print(f"\n[DONE] Seeding complete!")
    print(f"   High Risk:     {high_risk} ({high_risk/num_predictions*100:.1f}%)")
    print(f"   Moderate Risk: {moderate_risk} ({moderate_risk/num_predictions*100:.1f}%)")
    print(f"   Low Risk:      {low_risk} ({low_risk/num_predictions*100:.1f}%)")
    print(f"   Total Events:  {num_predictions + 3}")
    
    return {
        'predictions': num_predictions,
        'high_risk': high_risk,
        'moderate_risk': moderate_risk,
        'low_risk': low_risk
    }


if __name__ == "__main__":
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    seed_synthetic_data(num)
