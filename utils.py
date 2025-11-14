"""
Utility functions for SUQ-3 pipeline
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

class ModelEvaluator:
    """Evaluate model performance"""
    
    @staticmethod
    def evaluate(model, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            model: Keras model
            X_test, y_test: Test data
            
        Returns:
            dict: Metrics (accuracy, precision, recall, f1)
        """
        preds = model.predict(X_test, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='macro')
        }

class SparsityAnalyzer:
    """Analyze model sparsity"""
    
    @staticmethod
    def count_nonzeros(model):
        """
        Count non-zero weights per layer
        
        Args:
            model: Keras model
            
        Returns:
            pd.DataFrame: Sparsity statistics
        """
        rows = []
        for layer in model.layers:
            ws = layer.get_weights()
            if not ws:
                continue
            
            total = 0
            nonzero = 0
            for w in ws:
                total += w.size
                nonzero += np.count_nonzero(w)
            
            rows.append({
                'layer': layer.name,
                'nonzero': int(nonzero),
                'total': int(total),
                'sparsity': 1 - (nonzero / total) if total > 0 else 0
            })
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def model_size_kb(model):
        """Calculate model size in KB"""
        total_params = 0
        for layer in model.layers:
            ws = layer.get_weights()
            for w in ws:
                total_params += w.size
        
        # Approximate size in KB (float32 = 4 bytes per param)
        return (total_params * 4) / 1024

class ResultsComparator:
    """Compare results across pipeline stages"""
    
    @staticmethod
    def create_comparison_table(results_dict):
        """
        Create comprehensive comparison table
        
        Args:
            results_dict: {stage_name: {metrics_dict}}
            
        Returns:
            pd.DataFrame: Comparison table
        """
        data = []
        for stage_name, metrics in results_dict.items():
            row = {'Stage': stage_name}
            row.update(metrics)
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    @staticmethod
    def plot_comparison(results_df, output_path='results/comparison.png'):
        """Plot comparison results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy
        axes[0, 0].bar(results_df['Stage'], results_df['Accuracy'])
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim([0.9, 1.0])
        
        # Sparsity
        axes[0, 1].bar(results_df['Stage'], results_df['Sparsity'])
        axes[0, 1].set_title('Sparsity Comparison')
        axes[0, 1].set_ylabel('Sparsity')
        
        # Model Size
        axes[1, 0].bar(results_df['Stage'], results_df['Size (KB)'])
        axes[1, 0].set_title('Model Size')
        axes[1, 0].set_ylabel('Size (KB)')
        
        # F1 Score
        if 'F1' in results_df.columns:
            axes[1, 1].bar(results_df['Stage'], results_df['F1'])
            axes[1, 1].set_title('F1 Score')
            axes[1, 1].set_ylabel('F1')
            axes[1, 1].set_ylim([0.9, 1.0])
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Comparison plot saved to {output_path}")

class ReproducibilityHelper:
    """Ensure reproducibility"""
    
    @staticmethod
    def set_seeds(seed=42):
        """Set random seeds for reproducibility"""
        import random
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        print(f"Random seeds set to {seed}")
