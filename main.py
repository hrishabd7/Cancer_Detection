#!/usr/bin/env python3
"""
Cancer Detection Main Script

This script demonstrates the basic usage of the cancer detection system.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processing import load_data, preprocess_data, split_data
from src.models import CancerDetectionModel
from src.utils import setup_directories, calculate_metrics, log_experiment

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the cancer detection pipeline."""
    logger.info("Starting Cancer Detection System")
    
    # Setup project directories
    setup_directories()
    
    # Demo with synthetic data (replace with real data loading)
    logger.info("Creating synthetic demo data...")
    
    # Create synthetic dataset for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate synthetic features
    X = np.random.randn(n_samples, n_features)
    
    # Generate synthetic labels (0: benign, 1: malignant)
    # Add some correlation to make it realistic
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    data = pd.DataFrame(X, columns=feature_names)
    data['target'] = y
    
    logger.info(f"Synthetic dataset created: {data.shape}")
    logger.info(f"Class distribution: {data['target'].value_counts().to_dict()}")
    
    # Preprocess data
    processed_data = preprocess_data(data)
    
    # Split data
    train_data, test_data = split_data(processed_data, test_size=0.2)
    
    # Prepare features and labels
    X_train = train_data[feature_names].values
    y_train = train_data['target'].values
    X_test = test_data[feature_names].values
    y_test = test_data['target'].values
    
    # Train models
    models_to_test = ['random_forest', 'logistic_regression', 'svm']
    results = {}
    
    for model_type in models_to_test:
        logger.info(f"Training {model_type} model...")
        
        # Initialize and train model
        model = CancerDetectionModel(model_type=model_type)
        model.train(X_train, y_train)
        
        # Evaluate model
        metrics = model.evaluate(X_test, y_test)
        results[model_type] = metrics
        
        # Calculate additional metrics
        predictions = model.predict(X_test)
        detailed_metrics = calculate_metrics(y_test, predictions)
        
        # Log experiment
        experiment_config = {
            'model_type': model_type,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_features': n_features
        }
        
        log_experiment(f"{model_type}_experiment", detailed_metrics, experiment_config)
        
        # Save model
        model_path = f"models/saved/{model_type}_model.joblib"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_model(model_path)
        
        logger.info(f"{model_type} - Accuracy: {metrics['accuracy']:.4f}")
    
    # Display results summary
    logger.info("\n" + "="*50)
    logger.info("RESULTS SUMMARY")
    logger.info("="*50)
    
    for model_type, metrics in results.items():
        logger.info(f"{model_type.upper()}: Accuracy = {metrics['accuracy']:.4f}")
    
    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
    logger.info(f"\nBest performing model: {best_model.upper()}")
    logger.info(f"Best accuracy: {results[best_model]['accuracy']:.4f}")
    
    logger.info("Cancer Detection System completed successfully!")


if __name__ == "__main__":
    main()