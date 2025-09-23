"""
Data Processing Module

This module contains functions for preprocessing medical data and images
for cancer detection models.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from file.
    
    Args:
        file_path (str): Path to the data file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset for training.
    
    Args:
        data (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    # Basic preprocessing steps
    processed_data = data.copy()
    
    # Remove missing values
    processed_data = processed_data.dropna()
    
    # Basic data cleaning
    processed_data = processed_data.reset_index(drop=True)
    
    logger.info(f"Data preprocessed: {len(processed_data)} samples")
    return processed_data


def split_data(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets.
    
    Args:
        data (pd.DataFrame): Dataset to split
        test_size (float): Proportion of test data
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and testing datasets
    """
    from sklearn.model_selection import train_test_split
    
    train_data, test_data = train_test_split(
        data, 
        test_size=test_size, 
        random_state=random_state,
        stratify=data.iloc[:, -1] if len(data.columns) > 1 else None
    )
    
    logger.info(f"Data split: {len(train_data)} train, {len(test_data)} test samples")
    return train_data, test_data