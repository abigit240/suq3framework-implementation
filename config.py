# Configuration Settings for SUQ-3 Compression Pipeline

# Input and Output Directories
INPUT_DIR = '/path/to/input'
OUTPUT_DIR = '/path/to/output'
DATASET_PATH = 'Crop_recommendation.csv'
INPUT_DIM = 7  # Number of input features
N_CLASSES = 22  # Number of output classes
RESULTS_DIR = './results'


# Compression Settings
COMPRESSION_LEVEL = 5  # Range: 1 (fastest) to 9 (best compression)
ENABLE_LOGGING = True
LOG_LEVEL = 'INFO'  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# SUQ-3 Specific Settings
SEED = 42
BLOCK_SIZE = 4096  # Size of data blocks to be processed
MAX_CYCLES = 10    # Maximum number of cycles for compression
BASE_EPOCHS = 50  # Number of epochs for baseline model training
LEARNING_RATE = 0.001
GRU_UNITS = 64
DENSE_1_UNITS = 32
BATCH_SIZE = 128
# Model Hyperparameters
HYPERPARAMETERS = {
    'dropout_rate': 0.2,
    'weight_decay': 1e-4,
}
PRUNE_UNSTR_FT_EPOCHS = 10
PRUNE_UNSTR_FT_EPOCHS_FINAL = 20
BASE_EPOCHS_FINAL = 100
QAT_EPOCHS = 30
TRAIN_RATIO = 0.6
TEST_RATIO = 0.12
VAL_RATIO = 0.28
MODELS_DIR = './models'
STRUCTURED_PRUNING_TARGET = 0.5  # 50% sparsity 
UNSTRUCTURED_PRUNING_TARGET = 0.8  # 80% sparsity
STRUCTURED_PRUNING_TARGET_FINAL = 0.7  # 70% sparsity 
UNSTRUCTURED_PRUNING_TARGET_FINAL = 0.9  # 90% sparsity
STRUCTURED_PRUNING_CONFIG = {
    'pruning_type': 'l1_unstructured',
    'block_size': (1, 4),       # Pruning block size
    'initial_sparsity': 0.0,    # Start with no pruning
    'final_sparsity': 0.5,       # End with 50% sparsity
    'begin_step': 0,
    'end_step': 1000,
    'block_pooling_type': 'AVG'
}
UNSTRUCTURED_PRUNING_CONFIG = {
    'pruning_type': 'magnitude_unstructured',
    'initial_sparsity': 0.0,    # Start with no pruning
    'final_sparsity': 0.8,       # End with 80% sparsity
    'begin_step': 0,
    'end_step': 1000,
}        
# Other Configuration
VERSION = '1.0.0'
AUTHOR = 'abigit240'
DESCRIPTION = 'Configuration for SUQ-3 Compression Pipeline'