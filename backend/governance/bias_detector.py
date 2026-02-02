"""Bias Detection Module for AI Governance."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class BiasMetrics:
    """Container for bias detection metrics."""
    demographic_parity: float
    equalized_odds_tpr: float
    equalized_odds_fpr: float
    disparate_impact: float
    group_name: str
    reference_group: str
    sample_size_group: int
    sample_size_reference: int
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def is_biased(self, threshold: float = 0.1) -> bool:
        """Check if metrics indicate bias above threshold."""
        return (
            abs(self.demographic_parity) > threshold or
            abs(self.equalized_odds_tpr) > threshold or
            abs(self.equalized_odds_fpr) > threshold or
            self.disparate_impact < 0.8 or
            self.disparate_impact > 1.25
        )


class BiasDetector:
    """
    Comprehensive bias detection for ML models.
    
    Implements multiple fairness metrics:
    - Demographic Parity: P(Y_hat=1|A=a) = P(Y_hat=1|A=b)
    - Equalized Odds: P(Y_hat=1|A=a,Y=y) = P(Y_hat=1|A=b,Y=y)
    - Disparate Impact: P(Y_hat=1|A=a) / P(Y_hat=1|A=b) >= 0.8
    - Predictive Parity: P(Y=1|Y_hat=1,A=a) = P(Y=1|Y_hat=1,A=b)
    """
    
    def __init__(self, protected_attributes: List[str] = None):
        """
        Initialize bias detector.
        
        Args:
            protected_attributes: List of protected attribute column names
        """
        self.protected_attributes = protected_attributes or ['gender', 'age_group', 'ethnicity']
        self.bias_reports: List[Dict] = []
    
    def calculate_demographic_parity(
        self,
        predictions: np.ndarray,
        protected_attr: np.ndarray,
        reference_group: Any = None
    ) -> Dict[str, float]:
        """
        Calculate demographic parity difference.
        
        Demographic parity requires that positive prediction rates
        are equal across all groups.
        
        Returns:
            Dictionary of group -> parity difference from reference
        """
        unique_groups = np.unique(protected_attr)
        
        if reference_group is None:
            reference_group = unique_groups[0]
        
        ref_mask = protected_attr == reference_group
        ref_rate = predictions[ref_mask].mean()
        
        results = {}
        for group in unique_groups:
            group_mask = protected_attr == group
            group_rate = predictions[group_mask].mean()
            results[str(group)] = group_rate - ref_rate
        
        return results
    
    def calculate_equalized_odds(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        protected_attr: np.ndarray,
        reference_group: Any = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate equalized odds metrics.
        
        Equalized odds requires equal true positive rates and
        false positive rates across groups.
        
        Returns:
            Dictionary with TPR and FPR differences for each group
        """
        unique_groups = np.unique(protected_attr)
        
        if reference_group is None:
            reference_group = unique_groups[0]
        
        def calculate_rates(preds, labs):
            positives = labs == 1
            negatives = labs == 0
            
            tpr = preds[positives].mean() if positives.sum() > 0 else 0
            fpr = preds[negatives].mean() if negatives.sum() > 0 else 0
            
            return tpr, fpr
        
        ref_mask = protected_attr == reference_group
        ref_tpr, ref_fpr = calculate_rates(predictions[ref_mask], labels[ref_mask])
        
        results = {}
        for group in unique_groups:
            group_mask = protected_attr == group
            group_tpr, group_fpr = calculate_rates(
                predictions[group_mask], labels[group_mask]
            )
            results[str(group)] = {
                'tpr_difference': group_tpr - ref_tpr,
                'fpr_difference': group_fpr - ref_fpr
            }
        
        return results
    
    def calculate_disparate_impact(
        self,
        predictions: np.ndarray,
        protected_attr: np.ndarray,
        reference_group: Any = None
    ) -> Dict[str, float]:
        """
        Calculate disparate impact ratio.
        
        The 4/5ths (80%) rule: selection rate for any group
        should be at least 80% of the rate for the highest group.
        
        Returns:
            Dictionary of group -> impact ratio relative to reference
        """
        unique_groups = np.unique(protected_attr)
        
        if reference_group is None:
            reference_group = unique_groups[0]
        
        ref_mask = protected_attr == reference_group
        ref_rate = predictions[ref_mask].mean()
        
        if ref_rate == 0:
            ref_rate = 1e-10  # Avoid division by zero
        
        results = {}
        for group in unique_groups:
            group_mask = protected_attr == group
            group_rate = predictions[group_mask].mean()
            results[str(group)] = group_rate / ref_rate
        
        return results
    
    def calculate_predictive_parity(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        protected_attr: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate predictive parity (positive predictive value parity).
        
        Requires equal precision across groups.
        
        Returns:
            Dictionary of group -> PPV (precision)
        """
        unique_groups = np.unique(protected_attr)
        
        results = {}
        for group in unique_groups:
            group_mask = protected_attr == group
            group_preds = predictions[group_mask]
            group_labels = labels[group_mask]
            
            positive_preds = group_preds == 1
            if positive_preds.sum() > 0:
                ppv = group_labels[positive_preds].mean()
            else:
                ppv = 0
            
            results[str(group)] = ppv
        
        return results
    
    def run_full_analysis(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        protected_attr: np.ndarray,
        attribute_name: str,
        reference_group: Any = None
    ) -> Dict[str, BiasMetrics]:
        """
        Run complete bias analysis for a protected attribute.
        
        Returns:
            Dictionary of group -> BiasMetrics
        """
        binary_predictions = (predictions >= 0.5).astype(int)
        
        unique_groups = np.unique(protected_attr)
        if reference_group is None:
            reference_group = unique_groups[0]
        
        dp = self.calculate_demographic_parity(binary_predictions, protected_attr, reference_group)
        eo = self.calculate_equalized_odds(binary_predictions, labels, protected_attr, reference_group)
        di = self.calculate_disparate_impact(binary_predictions, protected_attr, reference_group)
        
        results = {}
        for group in unique_groups:
            group_str = str(group)
            ref_str = str(reference_group)
            
            group_mask = protected_attr == group
            ref_mask = protected_attr == reference_group
            
            metrics = BiasMetrics(
                demographic_parity=dp[group_str],
                equalized_odds_tpr=eo[group_str]['tpr_difference'],
                equalized_odds_fpr=eo[group_str]['fpr_difference'],
                disparate_impact=di[group_str],
                group_name=group_str,
                reference_group=ref_str,
                sample_size_group=group_mask.sum(),
                sample_size_reference=ref_mask.sum()
            )
            results[group_str] = metrics
        
        # Log report
        report = {
            'attribute': attribute_name,
            'reference_group': str(reference_group),
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {k: v.to_dict() for k, v in results.items()}
        }
        self.bias_reports.append(report)
        
        return results
    
    def generate_bias_report(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        data: pd.DataFrame,
        protected_columns: List[str] = None
    ) -> Dict:
        """
        Generate comprehensive bias report for multiple protected attributes.
        
        Returns:
            Full bias analysis report
        """
        if protected_columns is None:
            protected_columns = [col for col in self.protected_attributes if col in data.columns]
        
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'total_samples': len(predictions),
            'positive_rate': float(labels.mean()),
            'predicted_positive_rate': float((predictions >= 0.5).mean()),
            'attributes': {},
            'overall_bias_detected': False,
            'bias_summary': []
        }
        
        for col in protected_columns:
            if col not in data.columns:
                logger.warning(f"Protected attribute '{col}' not found in data")
                continue
            
            protected_attr = data[col].values
            
            # Skip if too few unique values
            unique_values = np.unique(protected_attr[~pd.isna(protected_attr)])
            if len(unique_values) < 2:
                continue
            
            # Run analysis
            metrics = self.run_full_analysis(
                predictions, labels, protected_attr, col
            )
            
            # Check for bias
            attribute_report = {
                'unique_groups': len(unique_values),
                'groups': [str(v) for v in unique_values],
                'metrics': {}
            }
            
            for group, metric in metrics.items():
                attribute_report['metrics'][group] = metric.to_dict()
                
                if metric.is_biased():
                    report['overall_bias_detected'] = True
                    report['bias_summary'].append({
                        'attribute': col,
                        'group': group,
                        'issue': self._describe_bias(metric)
                    })
            
            report['attributes'][col] = attribute_report
        
        return report
    
    def _describe_bias(self, metrics: BiasMetrics) -> str:
        """Generate human-readable description of detected bias."""
        issues = []
        
        if abs(metrics.demographic_parity) > 0.1:
            direction = "higher" if metrics.demographic_parity > 0 else "lower"
            issues.append(f"Prediction rate {direction} by {abs(metrics.demographic_parity):.2%}")
        
        if abs(metrics.equalized_odds_tpr) > 0.1:
            issues.append(f"True positive rate difference: {metrics.equalized_odds_tpr:.2%}")
        
        if abs(metrics.equalized_odds_fpr) > 0.1:
            issues.append(f"False positive rate difference: {metrics.equalized_odds_fpr:.2%}")
        
        if metrics.disparate_impact < 0.8:
            issues.append(f"Disparate impact ratio: {metrics.disparate_impact:.2f} (below 0.8 threshold)")
        elif metrics.disparate_impact > 1.25:
            issues.append(f"Disparate impact ratio: {metrics.disparate_impact:.2f} (above 1.25 threshold)")
        
        return "; ".join(issues) if issues else "No significant bias detected"
    
    def get_mitigation_recommendations(self, report: Dict) -> List[Dict]:
        """
        Generate recommendations for bias mitigation.
        
        Returns:
            List of mitigation recommendations
        """
        recommendations = []
        
        for summary in report.get('bias_summary', []):
            attribute = summary['attribute']
            group = summary['group']
            issue = summary['issue']
            
            rec = {
                'attribute': attribute,
                'group': group,
                'issue': issue,
                'recommendations': []
            }
            
            if 'Prediction rate' in issue:
                rec['recommendations'].extend([
                    "Consider reweighting training samples",
                    "Apply threshold adjustment for this group",
                    "Review feature engineering for proxy variables"
                ])
            
            if 'True positive rate' in issue:
                rec['recommendations'].extend([
                    "Examine if group has different baseline characteristics",
                    "Consider calibration adjustments",
                    "Review for missing or biased features"
                ])
            
            if 'Disparate impact' in issue:
                rec['recommendations'].extend([
                    "Legal/compliance review recommended",
                    "Consider adversarial debiasing techniques",
                    "Audit training data for historical bias"
                ])
            
            recommendations.append(rec)
        
        return recommendations
