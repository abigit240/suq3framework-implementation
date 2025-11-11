# Configuration Settings for SUQ-3 Compression Pipeline

# Input and Output Directories
INPUT_DIR = '/path/to/input'
OUTPUT_DIR = '/path/to/output'

# Compression Settings
COMPRESSION_LEVEL = 5  # Range: 1 (fastest) to 9 (best compression)
ENABLE_LOGGING = True
LOG_LEVEL = 'INFO'  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# SUQ-3 Specific Settings
BLOCK_SIZE = 4096  # Size of data blocks to be processed
MAX_CYCLES = 10    # Maximum number of cycles for compression

# Other Configuration
VERSION = '1.0.0'
AUTHOR = 'abigit240'
DESCRIPTION = 'Configuration for SUQ-3 Compression Pipeline'