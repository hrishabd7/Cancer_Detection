"""
Models Module

This module contains machine learning models for cancer detection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Tuple, Optional, Any
import joblib
import logging

logger = logging.getLogger(__name__)


class CancerDetectionModel:
    """
    Base class for cancer detection models.
    """
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize the cancer detection model.
        
        Args:
            model_type (str): Type of model to use ('random_forest', 'logistic_regression', 'svm')
        """
        self.model_type = model_type
        self.model = self._initialize_model()
        self.is_trained = False
        
    def _initialize_model(self) -> Any:
        """Initialize the specified model."""
        if self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "logistic_regression":
            return LogisticRegression(random_state=42)
        elif self.model_type == "svm":
            return SVC(random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
        """
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Model training completed")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X_test (np.ndarray): Test features
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X_test)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the model performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        predictions = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'classification_report': classification_report(y_test, predictions),
            'confusion_matrix': confusion_matrix(y_test, predictions)
        }
        
        logger.info(f"Model accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    def save_model(self, file_path: str) -> None:
        """
        Save the trained model.
        
        Args:
            file_path (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        joblib.dump(self.model, file_path)
        logger.info(f"Model saved to {file_path}")
    
    def load_model(self, file_path: str) -> None:
        """
        Load a trained model.
        
        Args:
            file_path (str): Path to the saved model
        """
        self.model = joblib.load(file_path)
        self.is_trained = True
        logger.info(f"Model loaded from {file_path}")