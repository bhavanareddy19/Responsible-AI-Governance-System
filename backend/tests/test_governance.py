"""Test suite for governance modules."""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from governance.bias_detector import BiasDetector, BiasMetrics
from governance.fairness_metrics import FairnessMetrics
from governance.compliance_checker import ComplianceChecker


class TestBiasDetector:
    """Tests for BiasDetector."""
    
    def test_demographic_parity(self):
        """Test demographic parity calculation."""
        detector = BiasDetector()
        
        predictions = np.array([1, 1, 0, 0, 1, 0])
        protected = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
        
        result = detector.calculate_demographic_parity(predictions, protected)
        
        assert 'A' in result
        assert 'B' in result
    
    def test_disparate_impact(self):
        """Test disparate impact calculation."""
        detector = BiasDetector()
        
        # Equal rates should give ratio of 1
        predictions = np.array([1, 1, 0, 1, 1, 0])
        protected = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
        
        result = detector.calculate_disparate_impact(predictions, protected)
        
        assert all(v > 0 for v in result.values())


class TestFairnessMetrics:
    """Tests for FairnessMetrics."""
    
    def test_fairness_scorecard(self):
        """Test fairness scorecard generation."""
        fairness = FairnessMetrics()
        
        predictions = np.random.random(100)
        labels = np.random.randint(0, 2, 100)
        protected = np.random.choice(['A', 'B'], 100)
        
        scorecard = fairness.calculate_fairness_scorecard(predictions, labels, protected)
        
        assert 0 <= scorecard.overall_fairness_score <= 100
        assert isinstance(scorecard.passes_threshold, bool)


class TestComplianceChecker:
    """Tests for ComplianceChecker."""
    
    def test_hipaa_compliance(self):
        """Test HIPAA compliance checks."""
        checker = ComplianceChecker()
        
        model_config = {'model_version': '1.0.0'}
        data_handling = {
            'encryption_enabled': True,
            'rbac_enabled': True,
            'audit_logging': True,
            'retention_years': 7
        }
        
        checks = checker.check_hipaa_compliance(model_config, data_handling)
        
        assert len(checks) >= 4
        assert all(c.passed for c in checks)
    
    def test_full_compliance_check(self):
        """Test full compliance check."""
        checker = ComplianceChecker()
        
        report = checker.run_full_compliance_check(
            model_metadata={'model_type': 'HealthcareRiskModel', 'model_version': '1.0.0', 'created_at': '2024-01-01'},
            data_handling={'encryption_enabled': True, 'rbac_enabled': True, 'audit_logging': True, 'retention_years': 7},
            validation_results={'auc': 0.85, 'bias_tested': True}
        )
        
        assert report.report_id is not None
        assert 'compliance_percentage' in report.summary


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
