"""
Data loading and preprocessing utilities for agricultural dataset
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import config

class DataLoader:
    """Load and preprocess Crop Recommendation dataset"""
    
    def __init__(self, csv_path=config.DATASET_PATH):
        """
        Initialize DataLoader
        
        Args:
            csv_path: Path to Crop_recommendation.csv
        """
        self.csv_path = csv_path
        self.X_mean = None
        self.X_std = None
        self.label_map = None
        
    def load_and_split(self):
        """
        Load CSV and split into train/val/test sets
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Load CSV
        assert os.path.exists(self.csv_path), f"{self.csv_path} not found"
        df = pd.read_csv(self.csv_path)
        
        print(f'Loaded CSV with shape: {df.shape}')
        
        # Extract features and labels
        X = df.iloc[:, :-1].values.astype(np.float32)
        y_raw = df.iloc[:, -1].values
        
        # Encode labels
        labels, uniques = pd.factorize(y_raw)
        y = labels.astype(np.int32)
        self.label_map = uniques
        
        print(f'Detected classes: {len(uniques)} -> {list(uniques)}')
        
        # Normalize features
        self.X_mean = X.mean(axis=0, keepdims=True)
        self.X_std = X.std(axis=0, keepdims=True) + 1e-9
        X_norm = (X - self.X_mean) / self.X_std
        
        # Split: 60% train, 28% val, 12% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_norm, y, 
            test_size=1 - config.TRAIN_RATIO, 
            stratify=y, 
            random_state=config.SEED
        )
        
        val_frac = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, 
            test_size=1 - val_frac, 
            stratify=y_temp, 
            random_state=config.SEED
        )
        
        print(f'Splits: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}')
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def normalize(self, X):
        """Apply stored normalization to new data"""
        assert self.X_mean is not None, "Must call load_and_split first"
        return (X - self.X_mean) / self.X_std

import os
