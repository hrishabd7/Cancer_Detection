"""
Unit tests for the cancer detection project.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path for importing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import preprocess_data, split_data
from models import CancerDetectionModel
from utils import calculate_metrics


class TestDataProcessing(unittest.TestCase):
    """Test data processing functions."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        processed = preprocess_data(self.test_data)
        
        # Check that NaN values are removed
        self.assertFalse(processed.isnull().any().any())
        
        # Check that the shape is reduced (due to NaN removal)
        self.assertLess(len(processed), len(self.test_data))
    
    def test_split_data(self):
        """Test data splitting."""
        clean_data = self.test_data.dropna()
        train_data, test_data = split_data(clean_data, test_size=0.3)
        
        # Check that data is split correctly
        total_samples = len(clean_data)
        expected_train_size = int(total_samples * 0.7)
        expected_test_size = total_samples - expected_train_size
        
        self.assertEqual(len(train_data), expected_train_size)
        self.assertEqual(len(test_data), expected_test_size)


class TestCancerDetectionModel(unittest.TestCase):
    """Test cancer detection model."""
    
    def setUp(self):
        """Set up test data and model."""
        np.random.seed(42)
        self.X_train = np.random.randn(100, 5)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.randn(20, 5)
        self.y_test = np.random.randint(0, 2, 20)
        
        self.model = CancerDetectionModel(model_type="random_forest")
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertFalse(self.model.is_trained)
        self.assertEqual(self.model.model_type, "random_forest")
    
    def test_model_training(self):
        """Test model training."""
        self.model.train(self.X_train, self.y_train)
        self.assertTrue(self.model.is_trained)
    
    def test_model_prediction(self):
        """Test model prediction."""
        self.model.train(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.y_test))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        self.model.train(self.X_train, self.y_train)
        metrics = self.model.evaluate(self.X_test, self.y_test)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('classification_report', metrics)
        self.assertIn('confusion_matrix', metrics)
        
        # Accuracy should be between 0 and 1
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        
        # All metrics should be between 0 and 1
        for metric_value in metrics.values():
            self.assertGreaterEqual(metric_value, 0)
            self.assertLessEqual(metric_value, 1)


if __name__ == '__main__':
    unittest.main()