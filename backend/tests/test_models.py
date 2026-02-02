"""Test suite for ML models."""
import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.models import HealthcareRiskModel, create_default_model, ALL_FEATURES
from ml.training import TrainingPipeline, DataGenerator


class TestHealthcareRiskModel:
    """Tests for HealthcareRiskModel."""
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = create_default_model()
        assert model is not None
        assert model.input_dim == len(ALL_FEATURES)
    
    def test_forward_pass(self):
        """Test forward pass produces valid output."""
        model = create_default_model()
        model.eval()
        
        x = torch.randn(10, len(ALL_FEATURES))
        output = model(x)
        
        assert output.shape == (10, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)
    
    def test_predict_proba(self):
        """Test probability predictions."""
        model = create_default_model()
        x = torch.randn(5, len(ALL_FEATURES))
        
        probs = model.predict_proba(x)
        
        assert probs.shape == (5, 1)
        assert np.all(probs >= 0) and np.all(probs <= 1)
    
    def test_model_metadata(self):
        """Test model metadata generation."""
        model = create_default_model()
        metadata = model.get_model_metadata()
        
        assert 'model_type' in metadata
        assert 'model_version' in metadata
        assert 'architecture' in metadata
        assert metadata['architecture']['input_dim'] == len(ALL_FEATURES)


class TestDataGenerator:
    """Tests for DataGenerator."""
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        X, y, df = DataGenerator.generate_synthetic_data(n_samples=100)
        
        assert X.shape == (100, len(ALL_FEATURES))
        assert y.shape == (100,)
        assert len(df) == 100
    
    def test_data_ranges(self):
        """Test that generated data has reasonable ranges."""
        X, y, df = DataGenerator.generate_synthetic_data(n_samples=1000)
        
        # Age should be between 18 and 100
        assert df['age'].min() >= 18
        assert df['age'].max() <= 100
        
        # Labels should be binary
        assert set(y) <= {0.0, 1.0}


class TestTrainingPipeline:
    """Tests for TrainingPipeline."""
    
    def test_training_single_epoch(self):
        """Test training for a single epoch."""
        model = create_default_model()
        X, y, _ = DataGenerator.generate_synthetic_data(n_samples=100)
        
        pipeline = TrainingPipeline(model, epochs=1, batch_size=32)
        summary = pipeline.train(X, y)
        
        assert 'training_results' in summary
        assert summary['training_results']['epochs_completed'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
