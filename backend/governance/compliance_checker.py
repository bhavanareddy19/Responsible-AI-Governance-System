"""Compliance Checker for AI Governance."""
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ComplianceStandard(str, Enum):
    HIPAA = "hipaa"
    FDA_AIML = "fda_aiml"
    GDPR = "gdpr"
    ISO_42001 = "iso_42001"


class RiskCategory(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplianceCheck:
    check_id: str
    standard: ComplianceStandard
    requirement: str
    description: str
    passed: bool
    details: str
    remediation: Optional[str] = None


@dataclass
class ComplianceReport:
    report_id: str
    timestamp: str
    model_version: str
    overall_compliant: bool
    risk_category: RiskCategory
    checks: List[ComplianceCheck]
    summary: Dict
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'checks': [asdict(c) for c in self.checks]
        }


class ComplianceChecker:
    """Regulatory compliance engine for healthcare AI."""
    
    def __init__(self):
        self.checks_performed: List[ComplianceReport] = []
    
    def check_hipaa_compliance(self, model_config: Dict, data_handling: Dict) -> List[ComplianceCheck]:
        """Check HIPAA compliance requirements."""
        checks = []
        
        # Check data encryption
        checks.append(ComplianceCheck(
            check_id="hipaa_001", standard=ComplianceStandard.HIPAA,
            requirement="Data Encryption", description="PHI must be encrypted at rest and in transit",
            passed=data_handling.get('encryption_enabled', False),
            details="Encryption status verified",
            remediation="Enable AES-256 encryption for all PHI data" if not data_handling.get('encryption_enabled') else None
        ))
        
        # Check access controls
        checks.append(ComplianceCheck(
            check_id="hipaa_002", standard=ComplianceStandard.HIPAA,
            requirement="Access Controls", description="Role-based access control required",
            passed=data_handling.get('rbac_enabled', False),
            details="RBAC configuration verified",
            remediation="Implement role-based access control" if not data_handling.get('rbac_enabled') else None
        ))
        
        # Check audit logging
        checks.append(ComplianceCheck(
            check_id="hipaa_003", standard=ComplianceStandard.HIPAA,
            requirement="Audit Logging", description="All PHI access must be logged",
            passed=data_handling.get('audit_logging', True),
            details="Audit logging active",
            remediation=None
        ))
        
        # Check data retention
        retention_years = data_handling.get('retention_years', 7)
        checks.append(ComplianceCheck(
            check_id="hipaa_004", standard=ComplianceStandard.HIPAA,
            requirement="Data Retention", description="Records must be retained for minimum 6 years",
            passed=retention_years >= 6,
            details=f"Current retention: {retention_years} years",
            remediation=f"Increase retention to minimum 6 years" if retention_years < 6 else None
        ))
        
        return checks
    
    def check_fda_aiml_compliance(self, model_metadata: Dict, validation_results: Dict) -> List[ComplianceCheck]:
        """Check FDA AI/ML guidance compliance."""
        checks = []
        
        # Model documentation
        has_docs = all(k in model_metadata for k in ['model_type', 'model_version', 'created_at'])
        checks.append(ComplianceCheck(
            check_id="fda_001", standard=ComplianceStandard.FDA_AIML,
            requirement="Model Documentation", description="Complete model documentation required",
            passed=has_docs, details="Model metadata completeness verified",
            remediation="Complete model documentation with all required fields" if not has_docs else None
        ))
        
        # Validation testing
        val_auc = validation_results.get('auc', 0)
        checks.append(ComplianceCheck(
            check_id="fda_002", standard=ComplianceStandard.FDA_AIML,
            requirement="Clinical Validation", description="Model must meet minimum performance thresholds",
            passed=val_auc >= 0.7, details=f"Validation AUC: {val_auc:.3f}",
            remediation="Improve model performance to meet clinical standards" if val_auc < 0.7 else None
        ))
        
        # Bias testing
        bias_tested = validation_results.get('bias_tested', False)
        checks.append(ComplianceCheck(
            check_id="fda_003", standard=ComplianceStandard.FDA_AIML,
            requirement="Bias Assessment", description="Bias testing across demographic groups required",
            passed=bias_tested, details="Bias assessment status",
            remediation="Complete bias assessment across all demographic groups" if not bias_tested else None
        ))
        
        # Explainability
        has_explainability = model_metadata.get('explainability_available', True)
        checks.append(ComplianceCheck(
            check_id="fda_004", standard=ComplianceStandard.FDA_AIML,
            requirement="Explainability", description="Model decisions must be explainable",
            passed=has_explainability, details="SHAP-based explanations available",
            remediation=None
        ))
        
        return checks
    
    def determine_risk_category(self, checks: List[ComplianceCheck]) -> RiskCategory:
        """Determine overall risk category based on compliance checks."""
        failed_checks = [c for c in checks if not c.passed]
        critical_failures = [c for c in failed_checks if c.check_id in ['hipaa_001', 'hipaa_003', 'fda_002']]
        
        if critical_failures:
            return RiskCategory.CRITICAL
        elif len(failed_checks) > 3:
            return RiskCategory.HIGH
        elif len(failed_checks) > 0:
            return RiskCategory.MEDIUM
        return RiskCategory.LOW
    
    def run_full_compliance_check(self, model_metadata: Dict, data_handling: Dict,
                                   validation_results: Dict) -> ComplianceReport:
        """Run comprehensive compliance check."""
        import uuid
        
        all_checks = []
        all_checks.extend(self.check_hipaa_compliance(model_metadata, data_handling))
        all_checks.extend(self.check_fda_aiml_compliance(model_metadata, validation_results))
        
        passed = sum(1 for c in all_checks if c.passed)
        failed = len(all_checks) - passed
        risk = self.determine_risk_category(all_checks)
        
        report = ComplianceReport(
            report_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            model_version=model_metadata.get('model_version', 'unknown'),
            overall_compliant=failed == 0,
            risk_category=risk,
            checks=all_checks,
            summary={'total_checks': len(all_checks), 'passed': passed, 'failed': failed,
                    'compliance_percentage': round((passed / len(all_checks)) * 100, 1)}
        )
        
        self.checks_performed.append(report)
        return report
