"""SHAP-based Explainability Module."""
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """Container for prediction explanation."""
    prediction_id: str
    prediction_value: float
    base_value: float
    feature_names: List[str]
    feature_values: List[float]
    shap_values: List[float]
    top_positive_features: List[Dict]
    top_negative_features: List[Dict]
    explanation_text: str


class SHAPExplainer:
    """SHAP-based model interpretability for healthcare predictions."""
    
    def __init__(self, model, feature_names: List[str], background_data: Optional[np.ndarray] = None):
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self._explainer = None
        self._base_value = 0.5
    
    def _model_predict(self, X: np.ndarray) -> np.ndarray:
        """Wrapper for model prediction."""
        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(X)
            output = self.model(tensor)
        return output.numpy().flatten()
    
    def initialize_explainer(self, background_samples: int = 100) -> None:
        """Initialize SHAP explainer with background data."""
        try:
            import shap
            if self.background_data is not None:
                bg = self.background_data[:background_samples]
            else:
                bg = np.random.randn(background_samples, len(self.feature_names))
            self._explainer = shap.KernelExplainer(self._model_predict, bg)
            self._base_value = self._model_predict(bg).mean()
            logger.info("SHAP explainer initialized")
        except ImportError:
            logger.warning("SHAP not available, using fallback gradient-based explanations")
            self._explainer = None
    
    def explain_prediction(self, features: np.ndarray, prediction_id: str = "unknown") -> Explanation:
        """Generate SHAP explanation for a single prediction."""
        features = np.atleast_2d(features)
        prediction = self._model_predict(features)[0]
        
        if self._explainer is not None:
            try:
                shap_values = self._explainer.shap_values(features, nsamples=100)[0]
            except Exception as e:
                logger.warning(f"SHAP failed: {e}, using gradient-based fallback")
                shap_values = self._gradient_explain(features)[0]
        else:
            shap_values = self._gradient_explain(features)[0]
        
        # Get top features
        sorted_idx = np.argsort(np.abs(shap_values))[::-1]
        top_positive = [{'feature': self.feature_names[i], 'value': float(features[0, i]),
                        'contribution': float(shap_values[i])} 
                       for i in sorted_idx if shap_values[i] > 0][:5]
        top_negative = [{'feature': self.feature_names[i], 'value': float(features[0, i]),
                        'contribution': float(shap_values[i])} 
                       for i in sorted_idx if shap_values[i] < 0][:5]
        
        explanation_text = self._generate_explanation_text(prediction, top_positive, top_negative)
        
        return Explanation(
            prediction_id=prediction_id, prediction_value=float(prediction),
            base_value=float(self._base_value), feature_names=self.feature_names,
            feature_values=features[0].tolist(), shap_values=shap_values.tolist(),
            top_positive_features=top_positive, top_negative_features=top_negative,
            explanation_text=explanation_text
        )
    
    def _gradient_explain(self, features: np.ndarray) -> np.ndarray:
        """Fallback gradient-based explanation using eval mode with gradient tracking."""
        self.model.eval()
        # Duplicate single sample to avoid BatchNorm single-sample issue during gradient computation
        if features.shape[0] == 1:
            features_batch = np.repeat(features, 2, axis=0)
        else:
            features_batch = features

        tensor = torch.FloatTensor(features_batch)
        tensor.requires_grad = True

        # Enable gradient computation in eval mode
        with torch.enable_grad():
            output = self.model(tensor)
            output[0].backward()

        gradients = tensor.grad.numpy()[:features.shape[0]] * features
        return gradients
    
    def _generate_explanation_text(self, prediction: float, positive: List[Dict], 
                                   negative: List[Dict]) -> str:
        """Generate human-readable explanation."""
        risk_level = "HIGH" if prediction >= 0.7 else "MODERATE" if prediction >= 0.4 else "LOW"
        
        lines = [f"Risk Assessment: {risk_level} ({prediction:.1%} probability)"]
        
        if positive:
            lines.append("\nFactors increasing risk:")
            for f in positive[:3]:
                lines.append(f"  • {f['feature']}: {f['value']:.2f} (+{f['contribution']:.1%})")
        
        if negative:
            lines.append("\nFactors decreasing risk:")
            for f in negative[:3]:
                lines.append(f"  • {f['feature']}: {f['value']:.2f} ({f['contribution']:.1%})")
        
        return "\n".join(lines)
    
    def batch_explain(self, features: np.ndarray, prediction_ids: Optional[List[str]] = None) -> List[Explanation]:
        """Generate explanations for multiple predictions."""
        if prediction_ids is None:
            prediction_ids = [f"pred_{i}" for i in range(len(features))]
        return [self.explain_prediction(features[i:i+1], prediction_ids[i]) 
                for i in range(len(features))]


class ClinicalRationaleGenerator:
    """Generate clinical decision rationale from SHAP explanations."""
    
    FEATURE_DESCRIPTIONS = {
        'systolic_bp': 'Systolic blood pressure',
        'diastolic_bp': 'Diastolic blood pressure', 
        'heart_rate': 'Heart rate',
        'age': 'Patient age',
        'diabetes': 'Diabetes diagnosis',
        'hypertension': 'Hypertension diagnosis',
        'heart_disease': 'History of heart disease',
        'creatinine': 'Creatinine level',
        'hemoglobin': 'Hemoglobin level',
        'previous_admissions_30d': 'Hospital admissions in past 30 days'
    }
    
    @classmethod
    def generate_clinical_rationale(cls, explanation: Explanation) -> Dict:
        """Convert SHAP explanation to clinical rationale."""
        risk_level = "HIGH" if explanation.prediction_value >= 0.7 else \
                    "MODERATE" if explanation.prediction_value >= 0.4 else "LOW"
        
        contributing_factors = []
        for f in explanation.top_positive_features[:5]:
            desc = cls.FEATURE_DESCRIPTIONS.get(f['feature'], f['feature'])
            contributing_factors.append({
                'factor': desc, 'value': f['value'],
                'impact': 'High' if abs(f['contribution']) > 0.1 else 'Moderate',
                'direction': 'increasing risk'
            })
        
        protective_factors = []
        for f in explanation.top_negative_features[:5]:
            desc = cls.FEATURE_DESCRIPTIONS.get(f['feature'], f['feature'])
            protective_factors.append({
                'factor': desc, 'value': f['value'],
                'impact': 'High' if abs(f['contribution']) > 0.1 else 'Moderate',
                'direction': 'decreasing risk'
            })
        
        recommendations = cls._generate_recommendations(explanation, risk_level)
        
        return {
            'risk_assessment': {'level': risk_level, 'probability': explanation.prediction_value,
                               'confidence': 'High' if abs(explanation.prediction_value - 0.5) > 0.3 else 'Moderate'},
            'contributing_factors': contributing_factors,
            'protective_factors': protective_factors,
            'clinical_recommendations': recommendations,
            'disclaimer': "This AI-generated assessment is intended to support clinical decision-making. "
                         "Final decisions should be made by qualified healthcare professionals."
        }
    
    @classmethod
    def _generate_recommendations(cls, explanation: Explanation, risk_level: str) -> List[str]:
        """Generate clinical recommendations based on risk factors."""
        recommendations = []
        
        feature_map = dict(zip(explanation.feature_names, explanation.feature_values))
        
        if risk_level == "HIGH":
            recommendations.append("Consider immediate clinical review")
            recommendations.append("Enhanced monitoring recommended")
        
        if feature_map.get('systolic_bp', 0) > 140:
            recommendations.append("Blood pressure management review recommended")
        
        if feature_map.get('creatinine', 0) > 1.5:
            recommendations.append("Renal function assessment recommended")
        
        if feature_map.get('previous_admissions_30d', 0) > 0:
            recommendations.append("Review discharge planning and follow-up care")
        
        return recommendations if recommendations else ["Continue standard care protocols"]
