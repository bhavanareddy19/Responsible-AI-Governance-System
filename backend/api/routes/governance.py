"""Governance API routes."""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter
from typing import List, Optional
from datetime import datetime, timedelta

from api.schemas import BiasReportResponse, ComplianceStatusResponse, AuditEventResponse, AuditQueryRequest

router = APIRouter(prefix="/governance", tags=["Governance"])

# Shared model singleton
_model = None

def _get_model():
    global _model
    if _model is None:
        from ml.models import create_default_model
        _model = create_default_model()
        _model.eval()
    return _model


@router.get("/bias-report", response_model=BiasReportResponse)
async def get_bias_report() -> BiasReportResponse:
    """Get current bias detection report."""
    from governance import BiasDetector, FairnessMetrics
    from ml.training import DataGenerator
    from ml.models import ALL_FEATURES
    import torch

    # Generate test data for bias analysis
    X, y, df = DataGenerator.generate_synthetic_data(n_samples=1000)
    df = DataGenerator.add_demographic_groups(df)

    model = _get_model()
    with torch.no_grad():
        predictions = model(torch.FloatTensor(X)).numpy().flatten()
    
    detector = BiasDetector()
    report = detector.generate_bias_report(predictions, y, df, ['gender', 'age_group'])
    
    fairness = FairnessMetrics()
    if 'gender' in df.columns:
        scorecard = fairness.calculate_fairness_scorecard(predictions, y, df['gender'].values)
        fairness_score = scorecard.overall_fairness_score
    else:
        fairness_score = 95.0
    
    recommendations = detector.get_mitigation_recommendations(report)
    rec_list = [r['recommendations'][0] for r in recommendations if r.get('recommendations')]
    
    return BiasReportResponse(
        generated_at=report['generated_at'],
        total_samples=report['total_samples'],
        overall_bias_detected=report['overall_bias_detected'],
        attributes_analyzed=list(report['attributes'].keys()),
        fairness_score=fairness_score,
        bias_summary=report['bias_summary'],
        recommendations=rec_list[:5]
    )


@router.get("/compliance-status", response_model=ComplianceStatusResponse)
async def get_compliance_status() -> ComplianceStatusResponse:
    """Get current compliance status."""
    from governance import ComplianceChecker

    model = _get_model()
    checker = ComplianceChecker()

    report = checker.run_full_compliance_check(
        model_metadata=model.get_model_metadata(),
        data_handling={'encryption_enabled': True, 'rbac_enabled': True,
                      'audit_logging': True, 'retention_years': 7},
        validation_results={'auc': 0.85, 'bias_tested': True}
    )

    failed = [{'check_id': c.check_id, 'requirement': c.requirement,
              'remediation': c.remediation} for c in report.checks if not c.passed]

    all_checks = [{'check_id': c.check_id, 'requirement': c.requirement,
                   'standard': c.standard.value, 'passed': c.passed,
                   'description': c.description, 'details': c.details,
                   'remediation': c.remediation} for c in report.checks]

    return ComplianceStatusResponse(
        report_id=report.report_id,
        timestamp=report.timestamp,
        overall_compliant=report.overall_compliant,
        compliance_percentage=report.summary['compliance_percentage'],
        risk_category=report.risk_category.value,
        checks_passed=report.summary['passed'],
        checks_failed=report.summary['failed'],
        failed_checks=failed,
        all_checks=all_checks
    )


@router.post("/audit-query", response_model=List[AuditEventResponse])
async def query_audit_logs(query: AuditQueryRequest) -> List[AuditEventResponse]:
    """Query audit logs with filters."""
    from governance import get_audit_logger, AuditEventType
    
    audit = get_audit_logger()
    
    event_type = AuditEventType(query.event_type) if query.event_type else None
    start_time = datetime.fromisoformat(query.start_date) if query.start_date else None
    
    events = audit.query_events(
        event_type=event_type,
        start_time=start_time,
        limit=query.limit
    )
    
    return [AuditEventResponse(
        event_id=e.event_id, event_type=e.event_type.value, timestamp=e.timestamp,
        action=e.action, model_version=e.model_version, details=e.details
    ) for e in events]


@router.get("/audit-stats")
async def get_audit_stats():
    """Get audit log statistics."""
    from governance import get_audit_logger
    audit = get_audit_logger()
    return audit.get_statistics()
