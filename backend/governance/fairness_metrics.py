"""Fairness Metrics Calculator for AI Governance."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from sklearn.metrics import confusion_matrix
import logging

logger = logging.getLogger(__name__)


@dataclass
class FairnessScorecard:
    """Aggregated fairness scorecard."""
    overall_fairness_score: float  # 0-100
    demographic_parity_score: float
    equalized_odds_score: float
    calibration_score: float
    individual_fairness_score: float
    passes_threshold: bool
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict:
        return asdict(self)


class FairnessMetrics:
    """
    Comprehensive fairness metrics calculator.
    
    Implements both group and individual fairness metrics
    following established ML fairness literature.
    """
    
    def __init__(self, fairness_threshold: float = 0.1):
        """
        Initialize fairness metrics calculator.
        
        Args:
            fairness_threshold: Maximum acceptable disparity (default 10%)
        """
        self.fairness_threshold = fairness_threshold
        self.metrics_history: List[Dict] = []
    
    def statistical_parity_difference(
        self,
        predictions: np.ndarray,
        protected_attr: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate Statistical Parity Difference (SPD).
        
        SPD = P(Y_hat=1|protected=1) - P(Y_hat=1|protected=0)
        
        A value of 0 indicates perfect parity.
        """
        unique_groups = np.unique(protected_attr)
        rates = {}
        
        for group in unique_groups:
            mask = protected_attr == group
            rates[str(group)] = predictions[mask].mean()
        
        # Calculate pairwise differences
        groups = list(rates.keys())
        differences = {}
        
        for i, g1 in enumerate(groups):
            for g2 in groups[i+1:]:
                key = f"{g1}_vs_{g2}"
                differences[key] = rates[g1] - rates[g2]
        
        return {
            'rates': rates,
            'differences': differences,
            'max_difference': max(abs(v) for v in differences.values()) if differences else 0
        }
    
    def equal_opportunity_difference(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        protected_attr: np.ndarray
    ) -> Dict:
        """
        Calculate Equal Opportunity Difference.
        
        Measures difference in True Positive Rates across groups.
        EOD = TPR(protected=1) - TPR(protected=0)
        """
        unique_groups = np.unique(protected_attr)
        tprs = {}
        
        for group in unique_groups:
            mask = (protected_attr == group) & (labels == 1)
            if mask.sum() > 0:
                tprs[str(group)] = predictions[mask].mean()
            else:
                tprs[str(group)] = 0
        
        groups = list(tprs.keys())
        differences = {}
        
        for i, g1 in enumerate(groups):
            for g2 in groups[i+1:]:
                key = f"{g1}_vs_{g2}"
                differences[key] = tprs[g1] - tprs[g2]
        
        return {
            'tprs': tprs,
            'differences': differences,
            'max_difference': max(abs(v) for v in differences.values()) if differences else 0
        }
    
    def predictive_equality(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        protected_attr: np.ndarray
    ) -> Dict:
        """
        Calculate Predictive Equality.
        
        Measures difference in False Positive Rates across groups.
        """
        unique_groups = np.unique(protected_attr)
        fprs = {}
        
        for group in unique_groups:
            mask = (protected_attr == group) & (labels == 0)
            if mask.sum() > 0:
                fprs[str(group)] = predictions[mask].mean()
            else:
                fprs[str(group)] = 0
        
        groups = list(fprs.keys())
        differences = {}
        
        for i, g1 in enumerate(groups):
            for g2 in groups[i+1:]:
                key = f"{g1}_vs_{g2}"
                differences[key] = fprs[g1] - fprs[g2]
        
        return {
            'fprs': fprs,
            'differences': differences,
            'max_difference': max(abs(v) for v in differences.values()) if differences else 0
        }
    
    def calibration_by_group(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        protected_attr: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        """
        Calculate calibration metrics by group.
        
        Measures how well predicted probabilities match actual outcomes
        within each group.
        """
        unique_groups = np.unique(protected_attr)
        calibration = {}
        
        for group in unique_groups:
            mask = protected_attr == group
            group_preds = predictions[mask]
            group_labels = labels[mask]
            
            # Bin predictions
            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(group_preds, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            bin_calibration = []
            for i in range(n_bins):
                bin_mask = bin_indices == i
                if bin_mask.sum() > 0:
                    mean_pred = group_preds[bin_mask].mean()
                    mean_actual = group_labels[bin_mask].mean()
                    bin_calibration.append({
                        'bin': i,
                        'mean_predicted': float(mean_pred),
                        'mean_actual': float(mean_actual),
                        'count': int(bin_mask.sum()),
                        'calibration_error': float(abs(mean_pred - mean_actual))
                    })
            
            # Expected Calibration Error
            ece = sum(
                b['count'] * b['calibration_error'] 
                for b in bin_calibration
            ) / mask.sum() if mask.sum() > 0 else 0
            
            calibration[str(group)] = {
                'bins': bin_calibration,
                'expected_calibration_error': ece
            }
        
        return calibration
    
    def individual_fairness_score(
        self,
        features: np.ndarray,
        predictions: np.ndarray,
        k: int = 10
    ) -> Dict:
        """
        Calculate individual fairness metrics.
        
        Similar individuals should receive similar predictions.
        Uses k-nearest neighbors to find similar individuals.
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Fit KNN
        knn = NearestNeighbors(n_neighbors=k+1)  # +1 because point is its own neighbor
        knn.fit(features)
        
        # Find neighbors for each point
        distances, indices = knn.kneighbors(features)
        
        # Calculate prediction consistency for neighbors
        consistency_scores = []
        for i in range(len(predictions)):
            neighbor_preds = predictions[indices[i][1:]]  # Exclude self
            pred_variance = np.var(neighbor_preds)
            consistency_scores.append(1 - min(pred_variance, 1))
        
        return {
            'mean_consistency': float(np.mean(consistency_scores)),
            'min_consistency': float(np.min(consistency_scores)),
            'std_consistency': float(np.std(consistency_scores))
        }
    
    def counterfactual_fairness(
        self,
        model,
        features: np.ndarray,
        protected_idx: int
    ) -> Dict:
        """
        Calculate counterfactual fairness.
        
        Measures how predictions change when protected attribute is flipped.
        """
        import torch
        
        # Original predictions
        with torch.no_grad():
            original_tensor = torch.FloatTensor(features)
            original_preds = model(original_tensor).numpy()
        
        # Flip protected attribute
        counterfactual_features = features.copy()
        counterfactual_features[:, protected_idx] = 1 - counterfactual_features[:, protected_idx]
        
        # Counterfactual predictions
        with torch.no_grad():
            cf_tensor = torch.FloatTensor(counterfactual_features)
            cf_preds = model(cf_tensor).numpy()
        
        # Calculate changes
        pred_changes = np.abs(original_preds - cf_preds)
        
        return {
            'mean_prediction_change': float(pred_changes.mean()),
            'max_prediction_change': float(pred_changes.max()),
            'pct_significant_changes': float((pred_changes > 0.1).mean()),
            'counterfactually_fair': pred_changes.mean() < 0.05
        }
    
    def calculate_fairness_scorecard(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        protected_attr: np.ndarray,
        features: Optional[np.ndarray] = None
    ) -> FairnessScorecard:
        """
        Calculate comprehensive fairness scorecard.
        
        Returns aggregated fairness score and component scores.
        """
        # Calculate component metrics
        spd = self.statistical_parity_difference(predictions, protected_attr)
        eod = self.equal_opportunity_difference(predictions, labels, protected_attr)
        pe = self.predictive_equality(predictions, labels, protected_attr)
        cal = self.calibration_by_group(predictions, labels, protected_attr)
        
        # Convert to scores (100 = perfect fairness)
        dp_score = max(0, 100 - abs(spd['max_difference']) * 1000)
        eo_score = max(0, 100 - (abs(eod['max_difference']) + abs(pe['max_difference'])) * 500)
        
        # Calibration score
        max_ece = max(
            v['expected_calibration_error'] 
            for v in cal.values()
        )
        cal_score = max(0, 100 - max_ece * 500)
        
        # Individual fairness
        if features is not None:
            ind_fair = self.individual_fairness_score(features, predictions)
            ind_score = ind_fair['mean_consistency'] * 100
        else:
            ind_score = 100  # Assume perfect if not calculated
        
        # Overall score (weighted average)
        overall_score = (
            dp_score * 0.3 +
            eo_score * 0.3 +
            cal_score * 0.2 +
            ind_score * 0.2
        )
        
        passes = overall_score >= (100 - self.fairness_threshold * 100)
        
        scorecard = FairnessScorecard(
            overall_fairness_score=round(overall_score, 2),
            demographic_parity_score=round(dp_score, 2),
            equalized_odds_score=round(eo_score, 2),
            calibration_score=round(cal_score, 2),
            individual_fairness_score=round(ind_score, 2),
            passes_threshold=passes
        )
        
        # Store in history
        self.metrics_history.append({
            'scorecard': scorecard.to_dict(),
            'details': {
                'statistical_parity': spd,
                'equal_opportunity': eod,
                'predictive_equality': pe,
                'calibration': cal
            }
        })
        
        return scorecard
    
    def get_fairness_trends(self) -> Dict:
        """Get historical fairness trends."""
        if not self.metrics_history:
            return {'message': 'No historical data available'}
        
        scores = [h['scorecard']['overall_fairness_score'] for h in self.metrics_history]
        
        return {
            'total_evaluations': len(self.metrics_history),
            'current_score': scores[-1],
            'average_score': np.mean(scores),
            'score_trend': 'improving' if len(scores) > 1 and scores[-1] > scores[-2] else 'stable_or_declining',
            'min_score': min(scores),
            'max_score': max(scores)
        }
