"""
Main pipeline orchestration for SUQ-3 compression framework
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import modules
from data_loader import DataLoader
from baseline_model import BaselineModel
from pruning_stage1 import StructuredPruning
from pruning_stage2 import UnstructuredPruning
from quantization_stage3 import QuantizationAwareTraining
from utils import ModelEvaluator, SparsityAnalyzer, ResultsComparator, ReproducibilityHelper
import config

def main():
    """Run complete SUQ-3 compression pipeline"""
    
    # Set reproducibility
    ReproducibilityHelper.set_seeds(config.SEED)
    
    print("="*80)
    print("SUQ-3 Compression Pipeline for Agricultural Dataset")
    print("="*80)
    
    # ==================== DATA LOADING ====================
    print("\n[STEP 1] Loading and preprocessing data...")
    data_loader = DataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_and_split()
    
    # ==================== BASELINE MODEL ====================
    print("\n[STEP 2] Training baseline model...")
    baseline = BaselineModel()
    baseline.build()
    history_baseline = baseline.train(X_train, y_train, X_val, y_val)
    baseline.save(f'{config.MODELS_DIR}/baseline_model.h5')
    
    # Evaluate baseline
    baseline_metrics = ModelEvaluator.evaluate(baseline.model, X_test, y_test)
    baseline_size = SparsityAnalyzer.model_size_kb(baseline.model)
    baseline_sparsity_df = SparsityAnalyzer.count_nonzeros(baseline.model)
    
    print("\nBaseline Metrics:")
    print(f"  Accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {baseline_metrics['f1']:.4f}")
    print(f"  Model Size: {baseline_size:.2f} KB")
    print("\nBaseline Sparsity:")
    print(baseline_sparsity_df)
    
    # ==================== STAGE 1: STRUCTURED PRUNING ====================
    print("\n" + "="*80)
    print("[STAGE 1] Structured Block Pruning (Target: 61% sparsity)")
    print("="*80)
    
    stage1 = StructuredPruning(baseline, X_train)
    stage1.build_pruned_model()
    history_stage1 = stage1.train(X_train, y_train, X_val, y_val, epochs=20)
    stage1.strip_pruning()
    stage1.save(f'{config.MODELS_DIR}/stage1_structured_pruned.h5')
    
    # Evaluate Stage 1
    stage1_metrics = ModelEvaluator.evaluate(stage1.pruned_model, X_test, y_test)
    stage1_size = SparsityAnalyzer.model_size_kb(stage1.pruned_model)
    stage1_sparsity_df = SparsityAnalyzer.count_nonzeros(stage1.pruned_model)
    
    print("\nStage 1 Metrics:")
    print(f"  Accuracy: {stage1_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {stage1_metrics['f1']:.4f}")
    print(f"  Model Size: {stage1_size:.2f} KB")
    print(f"  Size Reduction: {((baseline_size - stage1_size) / baseline_size * 100):.2f}%")
    print("\nStage 1 Sparsity:")
    print(stage1_sparsity_df)
    
    # ==================== STAGE 2: UNSTRUCTURED PRUNING ====================
    print("\n" + "="*80)
    print("[STAGE 2] Unstructured Magnitude Pruning (Target: 76% sparsity)")
    print("="*80)
    
    stage2 = UnstructuredPruning(stage1.pruned_model, X_train)
    stage2.build_pruned_model()
    history_stage2 = stage2.train(X_train, y_train, X_val, y_val, epochs=config.PRUNE_UNSTR_FT_EPOCHS)
    stage2.strip_pruning()
    stage2.save(f'{config.MODELS_DIR}/stage2_unstructured_pruned.h5')
    
    # Evaluate Stage 2
    stage2_metrics = ModelEvaluator.evaluate(stage2.pruned_model, X_test, y_test)
    stage2_size = SparsityAnalyzer.model_size_kb(stage2.pruned_model)
    stage2_sparsity_df = SparsityAnalyzer.count_nonzeros(stage2.pruned_model)
    
    print("\nStage 2 Metrics:")
    print(f"  Accuracy: {stage2_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {stage2_metrics['f1']:.4f}")
    print(f"  Model Size: {stage2_size:.2f} KB")
    print(f"  Size Reduction from Baseline: {((baseline_size - stage2_size) / baseline_size * 100):.2f}%")
    print("\nStage 2 Sparsity:")
    print(stage2_sparsity_df)
    
    # ==================== STAGE 3: QAT & TFLITE ====================
    print("\n" + "="*80)
    print("[STAGE 3] Quantization-Aware Training & TFLite Export")
    print("="*80)
    
    stage3 = QuantizationAwareTraining(stage2.pruned_model, X_train)
    stage3.apply_qat()
    history_stage3 = stage3.train_qat(X_train, y_train, X_val, y_val, epochs=config.QAT_EPOCHS)
    stage3.save_qat_model(f'{config.MODELS_DIR}/stage3_qat_model.h5')
    
    # Convert to TFLite
    tflite_path = stage3.convert_to_tflite(f'{config.MODELS_DIR}/suq3_int8.tflite')
    
    # Evaluate Stage 3
    stage3_metrics = ModelEvaluator.evaluate(stage3.qat_model, X_test, y_test)
    stage3_size = SparsityAnalyzer.model_size_kb(stage3.qat_model)
    stage3_sparsity_df = SparsityAnalyzer.count_nonzeros(stage3.qat_model)
    
    print("\nStage 3 Metrics:")
    print(f"  Accuracy: {stage3_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {stage3_metrics['f1']:.4f}")
    print(f"  Model Size: {stage3_size:.2f} KB")
    
    # Benchmark TFLite
    tflite_latency = stage3.benchmark_tflite(tflite_path, X_test)
    
    # ==================== RESULTS SUMMARY ====================
    print("\n" + "="*80)
    print("PIPELINE RESULTS SUMMARY")
    print("="*80)
    
    results_dict = {
        'Baseline': {
            'Stage': 'Baseline',
            'Size (KB)': baseline_size,
            'Sparsity': 0.0,
            'Accuracy': baseline_metrics['accuracy'],
            'F1': baseline_metrics['f1'],
            'Latency (ms)': '-'
        },
        'Stage 1 (Structured)': {
            'Stage': 'Stage 1 (Structured)',
            'Size (KB)': stage1_size,
            'Sparsity': stage1_sparsity_df['sparsity'].mean(),
            'Accuracy': stage1_metrics['accuracy'],
            'F1': stage1_metrics['f1'],
            'Latency (ms)': '-'
        },
        'Stage 2 (Unstructured)': {
            'Stage': 'Stage 2 (Unstructured)',
            'Size (KB)': stage2_size,
            'Sparsity': stage2_sparsity_df['sparsity'].mean(),
            'Accuracy': stage2_metrics['accuracy'],
            'F1': stage2_metrics['f1'],
            'Latency (ms)': '-'
        },
        'Stage 3 (QAT+TFLite)': {
            'Stage': 'Stage 3 (QAT+TFLite)',
            'Size (KB)': stage3_size,
            'Sparsity': stage3_sparsity_df['sparsity'].mean(),
            'Accuracy': stage3_metrics['accuracy'],
            'F1': stage3_metrics['f1'],
            'Latency (ms)': f'{tflite_latency:.4f}'
        }
    }
    
    results_df = ResultsComparator.create_comparison_table(results_dict)
    print("\n" + results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv(f'{config.RESULTS_DIR}/pipeline_results.csv', index=False)
    print(f"\nResults saved to {config.RESULTS_DIR}/pipeline_results.csv")
    
    # Create comparison plot
    ResultsComparator.plot_comparison(results_df)
    
    print("\n" + "="*80)
    print("Pipeline execution completed successfully!")
    print("="*80)

if __name__ == '__main__':
    main()
