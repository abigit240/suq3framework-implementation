# SUQ-3 Framework Implementation

A complete Python implementation of the SUQ-3 (Structured → Unstructured → Quantization) compression pipeline for agricultural dataset classification on edge devices. SUQ-3 is a framework given by researchers by Vaiyapuri et al https://www.mdpi.com/2071-1050/17/12/5230

## Project Structure
suq3framework-implementation/
├── config.py # Configuration settings
├── data_loader.py # Dataset loading and preprocessing 
├── baseline_model.py # Baseline model architecture
├── pruning_stage1.py # Stage 1: Structured Pruning 
├── pruning_stage2.py # Stage 2: Unstructured Pruning 
├── quantization_stage3.py # Stage 3: QAT & TFLite Export
├── utils.py # Utility functions 
├── main.py # Pipeline orchestration 
├── requirements.txt # Dependencies 
└── README.md # This file

## Compression Pipeline

### Stage 1: Structured Block Pruning
- **Target Sparsity**: 61%
- **Method**: MxN block-level pruning using polynomial decay schedule
- **Output**: Structured pruned model

### Stage 2: Unstructured Magnitude Pruning
- **Target Sparsity**: 76% (cumulative)
- **Method**: Magnitude-based weight pruning
- **Input**: Stage 1 pruned model
- **Output**: Unstructured pruned model

### Stage 3: Quantization-Aware Training
- **Method**: INT8 quantization with TensorFlow Model Optimization
- **Export**: TFLite INT8 format for edge deployment
- **Input**: Stage 2 pruned model
- **Output**: Quantized TFLite model

## Model Architecture

- **Input**: 7 agricultural features
- **GRU Layer**: 48 units
- **Dense Layer 1**: 24 units (ReLU activation)
- **Dense Layer 2**: 22 units (Softmax - 22 crop classes)
- **Total Parameters**: 9,070

## Dataset

Kaggle Crop Recommendation Dataset
- **Samples**: 2,200
- **Classes**: 22 crop types
- **Train/Val/Test Split**: 60% / 28% / 12%

## Usage
Run the complete pipeline:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your dataset
# Copy Crop_recommendation.csv to ./data/Crop_recommendation.csv

# 3. Run the complete pipeline
python main.py
```
This will:
    Load and preprocess the agricultural dataset
    Train the baseline model
    Apply Stage 1 structured pruning
    Apply Stage 2 unstructured pruning
    Apply Stage 3 QAT and export to TFLite
    Evaluate all stages and create comparison report
## Expected Results
Stage	Size (KB)	Sparsity	Accuracy	F1 Score
Baseline	~36	0%	≥97%	≥0.97
Stage 1 (Structured)	~14	61%	≥97%	≥0.97
Stage 2 (Unstructured)	~9	76%	≥97%	≥0.97
Stage 3 (QAT+TFLite)	~8	76%	≥97%	≥0.97
## Performance Metrics
    Size Reduction: ≥2x from baseline
    Latency: Measured on TFLite INT8 model
    Accuracy: Maintained ≥97% across all stages
    Macro-F1: ≥0.96 maintained

## Output Files
    models/baseline_model.h5 - Baseline Keras model
    models/stage1_structured_pruned.h5 - Stage 1 pruned model
    models/stage2_unstructured_pruned.h5 - Stage 2 pruned model
    models/stage3_qat_model.h5 - Stage 3 QAT model
    models/suq3_int8.tflite - TFLite INT8 model
    results/pipeline_results.csv - Comprehensive results table
    results/comparison.png - Comparison visualization

## References
    Paper: SUQ-3 (Structured → Unstructured → Quantization)
    Dataset: Kaggle Crop Recommendation
    Tools:
        TensorFlow Model Optimization Toolkit
        TensorFlow Lite
        Scikit-learn
## Author
Implementation by abigit240 (2025)
## License
MIT License
