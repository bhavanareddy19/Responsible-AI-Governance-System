"""Training Pipeline for Healthcare Models."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
import json
import logging
from pathlib import Path

from ..models.healthcare_model import HealthcareRiskModel, ALL_FEATURES

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Automated training pipeline for healthcare ML models.
    
    Features:
    - Cross-validation training
    - Early stopping
    - Learning rate scheduling
    - Model checkpointing
    - Performance metrics logging
    - Governance-compliant audit trails
    """
    
    def __init__(
        self,
        model: HealthcareRiskModel,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        validation_split: float = 0.2,
        device: Optional[str] = None
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        
        # Training components
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Tracking
        self.training_history: List[Dict] = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.training_start_time: Optional[datetime] = None
        self.training_end_time: Optional[datetime] = None
    
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders."""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split into train/val
        val_size = int(len(dataset) * self.validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * batch_X.size(0)
            predicted = (outputs >= 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
        
        return {
            'loss': total_loss / total,
            'accuracy': correct / total
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item() * batch_X.size(0)
                predicted = (outputs >= 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
                
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(batch_y.cpu().numpy().flatten())
        
        # Calculate additional metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # AUC-ROC
        try:
            from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
            auc = roc_auc_score(all_labels, all_preds)
            binary_preds = (all_preds >= 0.5).astype(int)
            precision = precision_score(all_labels, binary_preds, zero_division=0)
            recall = recall_score(all_labels, binary_preds, zero_division=0)
            f1 = f1_score(all_labels, binary_preds, zero_division=0)
        except Exception:
            auc = 0.5
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        
        return {
            'loss': total_loss / total,
            'accuracy': correct / total,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict:
        """
        Full training loop with validation.
        
        Returns training summary including all metrics and governance data.
        """
        self.training_start_time = datetime.utcnow()
        logger.info(f"Starting training on {self.device}")
        
        train_loader, val_loader = self.prepare_data(X, y)
        
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Record history
            epoch_record = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_auc': val_metrics['auc'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1': val_metrics['f1'],
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'timestamp': datetime.utcnow().isoformat()
            }
            self.training_history.append(epoch_record)
            
            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val AUC: {val_metrics['auc']:.4f}"
                )
            
            # Early stopping check
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Execute callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch_record)
        
        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        self.training_end_time = datetime.utcnow()
        
        return self.get_training_summary()
    
    def get_training_summary(self) -> Dict:
        """Get training summary for governance logging."""
        return {
            'model_metadata': self.model.get_model_metadata(),
            'training_config': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'early_stopping_patience': self.early_stopping_patience,
                'validation_split': self.validation_split,
                'device': str(self.device)
            },
            'training_results': {
                'epochs_completed': len(self.training_history),
                'best_val_loss': self.best_val_loss,
                'final_metrics': self.training_history[-1] if self.training_history else None
            },
            'training_history': self.training_history,
            'timestamps': {
                'start': self.training_start_time.isoformat() if self.training_start_time else None,
                'end': self.training_end_time.isoformat() if self.training_end_time else None,
                'duration_seconds': (
                    (self.training_end_time - self.training_start_time).total_seconds()
                    if self.training_start_time and self.training_end_time else None
                )
            }
        }
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'training_summary': self.get_training_summary()
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")


class DataGenerator:
    """Generate synthetic healthcare data for testing and demonstration."""
    
    @staticmethod
    def generate_synthetic_data(
        n_samples: int = 10000,
        feature_names: List[str] = ALL_FEATURES,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Generate synthetic healthcare data with realistic distributions.
        
        Returns:
            X: Feature matrix
            y: Labels (risk scores)
            df: Full DataFrame with feature names
        """
        np.random.seed(seed)
        
        n_features = len(feature_names)
        X = np.zeros((n_samples, n_features))
        
        feature_idx = {name: i for i, name in enumerate(feature_names)}
        
        # Generate vitals
        if 'systolic_bp' in feature_idx:
            X[:, feature_idx['systolic_bp']] = np.random.normal(120, 20, n_samples)
        if 'diastolic_bp' in feature_idx:
            X[:, feature_idx['diastolic_bp']] = np.random.normal(80, 12, n_samples)
        if 'heart_rate' in feature_idx:
            X[:, feature_idx['heart_rate']] = np.random.normal(75, 15, n_samples)
        if 'respiratory_rate' in feature_idx:
            X[:, feature_idx['respiratory_rate']] = np.random.normal(16, 4, n_samples)
        if 'temperature' in feature_idx:
            X[:, feature_idx['temperature']] = np.random.normal(98.6, 1, n_samples)
        if 'oxygen_saturation' in feature_idx:
            X[:, feature_idx['oxygen_saturation']] = np.clip(np.random.normal(97, 3, n_samples), 80, 100)
        if 'pain_score' in feature_idx:
            X[:, feature_idx['pain_score']] = np.random.randint(0, 11, n_samples)
        
        # Generate demographics
        if 'age' in feature_idx:
            X[:, feature_idx['age']] = np.random.normal(55, 18, n_samples)
            X[:, feature_idx['age']] = np.clip(X[:, feature_idx['age']], 18, 100)
        if 'gender_male' in feature_idx:
            gender = np.random.binomial(1, 0.5, n_samples)
            X[:, feature_idx['gender_male']] = gender
            if 'gender_female' in feature_idx:
                X[:, feature_idx['gender_female']] = 1 - gender
        if 'bmi' in feature_idx:
            X[:, feature_idx['bmi']] = np.random.normal(27, 6, n_samples)
            X[:, feature_idx['bmi']] = np.clip(X[:, feature_idx['bmi']], 15, 50)
        
        # Generate lab values
        lab_means = {
            'hemoglobin': 13.5, 'white_blood_cells': 7.5, 'platelets': 250,
            'sodium': 140, 'potassium': 4.0, 'chloride': 100,
            'bicarbonate': 24, 'bun': 15, 'creatinine': 1.0,
            'glucose': 100, 'calcium': 9.5, 'magnesium': 2.0,
            'phosphate': 3.5, 'albumin': 4.0, 'bilirubin': 0.8,
            'alt': 25, 'ast': 25, 'alkaline_phosphatase': 70,
            'lactate': 1.0, 'troponin': 0.01
        }
        
        for lab, mean in lab_means.items():
            if lab in feature_idx:
                std = mean * 0.2
                X[:, feature_idx[lab]] = np.random.normal(mean, std, n_samples)
        
        # Generate medical history (binary)
        history_probs = {
            'diabetes': 0.15, 'hypertension': 0.30, 'heart_disease': 0.12,
            'copd': 0.08, 'ckd': 0.10, 'cancer': 0.07,
            'stroke_history': 0.05, 'icu_history': 0.10,
            'ventilator_history': 0.05, 'dialysis_history': 0.03
        }
        
        for condition, prob in history_probs.items():
            if condition in feature_idx:
                X[:, feature_idx[condition]] = np.random.binomial(1, prob, n_samples)
        
        # Generate count features
        if 'previous_admissions_30d' in feature_idx:
            X[:, feature_idx['previous_admissions_30d']] = np.random.poisson(0.3, n_samples)
        if 'previous_admissions_90d' in feature_idx:
            X[:, feature_idx['previous_admissions_90d']] = np.random.poisson(0.8, n_samples)
        if 'previous_admissions_365d' in feature_idx:
            X[:, feature_idx['previous_admissions_365d']] = np.random.poisson(2.0, n_samples)
        if 'current_medications_count' in feature_idx:
            X[:, feature_idx['current_medications_count']] = np.random.poisson(5, n_samples)
        if 'procedures_count' in feature_idx:
            X[:, feature_idx['procedures_count']] = np.random.poisson(1.5, n_samples)
        
        # Generate realistic labels based on features
        # Higher risk with: older age, more comorbidities, abnormal vitals
        risk_score = np.zeros(n_samples)
        
        if 'age' in feature_idx:
            risk_score += (X[:, feature_idx['age']] - 50) / 100
        if 'diabetes' in feature_idx:
            risk_score += X[:, feature_idx['diabetes']] * 0.2
        if 'heart_disease' in feature_idx:
            risk_score += X[:, feature_idx['heart_disease']] * 0.3
        if 'previous_admissions_30d' in feature_idx:
            risk_score += X[:, feature_idx['previous_admissions_30d']] * 0.15
        if 'creatinine' in feature_idx:
            risk_score += np.clip((X[:, feature_idx['creatinine']] - 1.2) / 2, 0, 0.3)
        
        # Add noise and convert to probability
        risk_score += np.random.normal(0, 0.1, n_samples)
        risk_prob = 1 / (1 + np.exp(-risk_score * 3))
        
        # Generate binary labels
        y = (np.random.random(n_samples) < risk_prob).astype(float)
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['label'] = y
        
        return X, y, df
    
    @staticmethod
    def add_demographic_groups(df: pd.DataFrame) -> pd.DataFrame:
        """Add demographic group columns for bias testing."""
        df = df.copy()
        
        # Age groups
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'],
                bins=[0, 40, 60, 80, 100],
                labels=['18-40', '41-60', '61-80', '80+']
            )
        
        # Gender
        if 'gender_male' in df.columns:
            df['gender'] = df['gender_male'].apply(lambda x: 'Male' if x == 1 else 'Female')
        
        return df
