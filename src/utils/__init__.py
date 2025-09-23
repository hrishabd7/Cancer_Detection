"""
Utility Functions

This module contains utility functions for the cancer detection project.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


def setup_directories(base_path: str = ".") -> None:
    """
    Create necessary directories for the project.
    
    Args:
        base_path (str): Base path for the project
    """
    directories = [
        os.path.join(base_path, "data", "raw"),
        os.path.join(base_path, "data", "processed"),
        os.path.join(base_path, "models", "saved"),
        os.path.join(base_path, "results"),
        os.path.join(base_path, "logs")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory created/verified: {directory}")


def save_config(config: Dict[str, Any], file_path: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        file_path (str): Path to save the configuration
    """
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved to {file_path}")


def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        file_path (str): Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(file_path, 'r') as f:
        config = json.load(f)
    logger.info(f"Configuration loaded from {file_path}")
    return config


def plot_distribution(data: pd.DataFrame, column: str, title: str = None) -> None:
    """
    Plot distribution of a column.
    
    Args:
        data (pd.DataFrame): Dataset
        column (str): Column to plot
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=column)
    plt.title(title or f"Distribution of {column}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(data: pd.DataFrame, figsize: tuple = (12, 8)) -> None:
    """
    Plot correlation matrix for numerical columns.
    
    Args:
        data (pd.DataFrame): Dataset
        figsize (tuple): Figure size
    """
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    correlation_matrix = data[numerical_cols].corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate various classification metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }


def log_experiment(experiment_name: str, metrics: Dict[str, float], config: Dict[str, Any]) -> None:
    """
    Log experiment results.
    
    Args:
        experiment_name (str): Name of the experiment
        metrics (Dict[str, float]): Experiment metrics
        config (Dict[str, Any]): Experiment configuration
    """
    log_entry = {
        'experiment_name': experiment_name,
        'metrics': metrics,
        'config': config,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    log_file = 'logs/experiments.json'
    
    # Load existing logs if file exists
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    # Append new log
    logs.append(log_entry)
    
    # Save updated logs
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4)
    
    logger.info(f"Experiment logged: {experiment_name}")