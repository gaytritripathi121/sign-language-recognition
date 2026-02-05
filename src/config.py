"""
Configuration file for ASL Alphabet Recognition System

This file contains all configuration parameters for the project including:
- Directory paths
- Dataset parameters
- Model architecture settings
- Training hyperparameters
- Data augmentation settings
- Inference parameters

Author: Your Name
Date: 2024
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Base directory - project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw' / 'asl_alphabet_train'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Model directories
MODEL_DIR = BASE_DIR / 'models' / 'saved_models'

# Results directories
RESULTS_DIR = BASE_DIR / 'results'

# Logs directory
LOGS_DIR = BASE_DIR / 'logs'

# Create directories if they don't exist
for directory in [DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET PARAMETERS
# ============================================================================

# Image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
IMG_CHANNELS = 3

# Number of classes in ASL alphabet dataset
NUM_CLASSES = 29

# Class names (A-Z, plus special characters)
CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

# Data split ratios (must sum to 1.0)
TRAIN_SPLIT = 0.7   # 70% for training
VAL_SPLIT = 0.15    # 15% for validation
TEST_SPLIT = 0.15   # 15% for testing

# Verify splits sum to 1.0
assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-6, \
    "Data splits must sum to 1.0"

# Batch size for training/validation/testing
BATCH_SIZE = 32  # Reduce to 16 if running out of memory on CPU

# ============================================================================
# MODEL ARCHITECTURE PARAMETERS
# ============================================================================

# Base model selection
# Options: 'MobileNetV2', 'EfficientNetB0'
BASE_MODEL = 'MobileNetV2'

# Transfer learning settings
FREEZE_BASE_LAYERS = True  # Freeze base model initially
FINE_TUNE_AT_LAYER = 100   # Layer index to start fine-tuning from

# Classification head architecture
DENSE_UNITS_1 = 512        # First dense layer units
DENSE_UNITS_2 = 256        # Second dense layer units
DROPOUT_RATE = 0.5         # Dropout rate for first dropout layer
DROPOUT_RATE_2 = 0.25      # Dropout rate for second dropout layer

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Training phases
INITIAL_EPOCHS = 15        # Number of epochs for initial training (frozen base)
FINE_TUNE_EPOCHS = 5     # Number of epochs for fine-tuning (unfrozen base)
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

# Learning rates
LEARNING_RATE = 0.001      # Initial learning rate
FINE_TUNE_LR = 0.0001      # Learning rate for fine-tuning phase

# Optimizer settings
OPTIMIZER = 'adam'         # Optimizer type
BETA_1 = 0.9              # Adam beta_1 parameter
BETA_2 = 0.999            # Adam beta_2 parameter
EPSILON = 1e-7            # Adam epsilon parameter

# Loss function
LOSS_FUNCTION = 'categorical_crossentropy'

# Metrics to track
METRICS = ['accuracy']

# ============================================================================
# CALLBACK PARAMETERS
# ============================================================================

# Model checkpoint settings
CHECKPOINT_MONITOR = 'val_accuracy'  # Metric to monitor
CHECKPOINT_MODE = 'max'              # 'max' for accuracy, 'min' for loss
SAVE_BEST_ONLY = True                # Save only the best model

# Early stopping settings
EARLY_STOPPING_MONITOR = 'val_loss'  # Metric to monitor
EARLY_STOPPING_PATIENCE = 5          # Number of epochs with no improvement
EARLY_STOPPING_MODE = 'min'          # 'min' for loss, 'max' for accuracy
RESTORE_BEST_WEIGHTS = True          # Restore best weights on early stop

# Reduce learning rate on plateau settings
REDUCE_LR_MONITOR = 'val_loss'       # Metric to monitor
REDUCE_LR_FACTOR = 0.5               # Factor to reduce LR by
REDUCE_LR_PATIENCE = 3               # Number of epochs with no improvement
REDUCE_LR_MIN_LR = 1e-7              # Minimum learning rate
REDUCE_LR_MODE = 'min'               # 'min' for loss, 'max' for accuracy

# TensorBoard settings
TENSORBOARD_HISTOGRAM_FREQ = 1       # Frequency to compute activation histograms
TENSORBOARD_WRITE_GRAPH = True       # Whether to visualize the graph
TENSORBOARD_WRITE_IMAGES = False     # Whether to write model weights as images

# ============================================================================
# DATA AUGMENTATION PARAMETERS
# ============================================================================

# Augmentation settings for training data
ROTATION_RANGE = 20           # Degree range for random rotations (±20°)
WIDTH_SHIFT_RANGE = 0.2       # Fraction of total width for horizontal shifts
HEIGHT_SHIFT_RANGE = 0.2      # Fraction of total height for vertical shifts
SHEAR_RANGE = 0.2             # Shear intensity
ZOOM_RANGE = 0.2              # Range for random zoom
HORIZONTAL_FLIP = True        # Randomly flip inputs horizontally
VERTICAL_FLIP = False         # Don't flip vertically (hands would be upside down)
BRIGHTNESS_RANGE = [0.8, 1.2] # Range for random brightness adjustment
FILL_MODE = 'nearest'         # Points outside boundaries filled by nearest pixel

# Rescaling (normalization)
RESCALE = 1./255              # Rescale pixel values to [0, 1]

# ============================================================================
# INFERENCE PARAMETERS
# ============================================================================

# Prediction settings
CONFIDENCE_THRESHOLD = 0.7    # Minimum confidence for predictions
TOP_K_PREDICTIONS = 3         # Number of top predictions to return

# Webcam demo settings
WEBCAM_ID = 0                 # Default webcam ID
WEBCAM_WIDTH = 640            # Webcam frame width
WEBCAM_HEIGHT = 480           # Webcam frame height
WEBCAM_FPS = 30               # Target FPS

# ROI (Region of Interest) settings for webcam
ROI_SIZE = 300                # Size of ROI box for hand detection
ROI_COLOR = (0, 255, 0)       # Green color for ROI box (BGR format)
ROI_THICKNESS = 3             # Thickness of ROI box border

# Display settings
FONT_FACE = 1                 # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.9              # Font size
FONT_THICKNESS = 2            # Font thickness
TEXT_COLOR = (255, 255, 255)  # White text color (BGR format)
PREDICTION_COLOR = (0, 255, 0) # Green for top prediction

# ============================================================================
# REPRODUCIBILITY SETTINGS
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Set random seeds
import random
import numpy as np
import tensorflow as tf

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# TensorFlow settings for reproducibility
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# ============================================================================
# KAGGLE DATASET INFORMATION
# ============================================================================

# Kaggle dataset identifier
KAGGLE_DATASET = 'grassknoted/asl-alphabet'
KAGGLE_DATASET_URL = 'https://www.kaggle.com/datasets/grassknoted/asl-alphabet'

# Dataset statistics (approximate)
TOTAL_IMAGES = 87000          # Total number of images in dataset
IMAGES_PER_CLASS = 3000       # Approximate images per class

# ============================================================================
# MODEL INFORMATION
# ============================================================================

# Model metadata
MODEL_NAME = 'ASL_Alphabet_Recognition'
MODEL_VERSION = '1.0.0'
MODEL_DESCRIPTION = 'Real-time ASL Alphabet Recognition using Transfer Learning'

# Pretrained weights
IMAGENET_WEIGHTS = 'imagenet'

# ============================================================================
# SYSTEM SETTINGS
# ============================================================================

# CPU/GPU settings
USE_GPU = False               # Set to True if GPU is available
if not USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

# Number of CPU cores to use
NUM_WORKERS = 4               # For data loading (multiprocessing)

# Memory settings
USE_MULTIPROCESSING = False   # Set to True for faster data loading (if enough RAM)
MAX_QUEUE_SIZE = 10          # Maximum size for generator queue

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

# Logging level
LOG_LEVEL = 'INFO'           # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'

# Verbose settings
VERBOSE_TRAINING = 1         # 0=silent, 1=progress bar, 2=one line per epoch
VERBOSE_PREDICTION = 0       # Verbosity for predictions

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Plot settings
PLOT_DPI = 300               # DPI for saved plots
PLOT_STYLE = 'seaborn'       # Matplotlib style
FIGURE_SIZE = (12, 8)        # Default figure size

# Confusion matrix settings
CM_FIGSIZE = (16, 14)        # Confusion matrix figure size
CM_CMAP = 'Blues'            # Colormap for confusion matrix
CM_ANNOT = True              # Show annotations on confusion matrix

# ============================================================================
# FILE PATHS (OUTPUTS)
# ============================================================================

# Model files
BEST_MODEL_PATH = MODEL_DIR / 'best_model.h5'
FINAL_MODEL_PATH = MODEL_DIR / 'final_model.h5'
MODEL_ARCHITECTURE_PATH = BASE_DIR / 'models' / 'model_architecture.png'

# Result files
TRAINING_HISTORY_PLOT = RESULTS_DIR / 'training_history.png'
TRAINING_HISTORY_JSON = RESULTS_DIR / 'training_history.json'
CONFUSION_MATRIX_PLOT = RESULTS_DIR / 'confusion_matrix.png'
CONFUSION_MATRIX_NORMALIZED = RESULTS_DIR / 'confusion_matrix_normalized.png'
CLASSIFICATION_REPORT_FILE = RESULTS_DIR / 'classification_report.txt'
MISCLASSIFICATION_ANALYSIS = RESULTS_DIR / 'misclassification_analysis.txt'
CLASS_DISTRIBUTION_PLOT = RESULTS_DIR / 'class_distribution.png'
SAMPLE_IMAGES_PLOT = RESULTS_DIR / 'sample_images.png'

# Dataset info
DATASET_INFO_JSON = DATA_DIR / 'dataset_info.json'

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_config():
    """Print current configuration"""
    print("="*70)
    print("ASL Alphabet Recognition - Configuration")
    print("="*70)
    print(f"\nProject Paths:")
    print(f"  Base Directory: {BASE_DIR}")
    print(f"  Data Directory: {DATA_DIR}")
    print(f"  Model Directory: {MODEL_DIR}")
    print(f"  Results Directory: {RESULTS_DIR}")
    
    print(f"\nDataset Parameters:")
    print(f"  Image Size: {IMG_SIZE}")
    print(f"  Number of Classes: {NUM_CLASSES}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Train Split: {TRAIN_SPLIT}")
    print(f"  Val Split: {VAL_SPLIT}")
    print(f"  Test Split: {TEST_SPLIT}")
    
    print(f"\nModel Parameters:")
    print(f"  Base Model: {BASE_MODEL}")
    print(f"  Freeze Base: {FREEZE_BASE_LAYERS}")
    print(f"  Fine-tune at Layer: {FINE_TUNE_AT_LAYER}")
    
    print(f"\nTraining Parameters:")
    print(f"  Initial Epochs: {INITIAL_EPOCHS}")
    print(f"  Fine-tune Epochs: {FINE_TUNE_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Fine-tune LR: {FINE_TUNE_LR}")
    print(f"  Dropout Rate: {DROPOUT_RATE}")
    
    print(f"\nData Augmentation:")
    print(f"  Rotation Range: ±{ROTATION_RANGE}°")
    print(f"  Horizontal Flip: {HORIZONTAL_FLIP}")
    print(f"  Brightness Range: {BRIGHTNESS_RANGE}")
    
    print(f"\nSystem Settings:")
    print(f"  Use GPU: {USE_GPU}")
    print(f"  Random Seed: {RANDOM_SEED}")
    print("="*70)


def verify_paths():
    """Verify all necessary directories exist"""
    required_dirs = [DATA_DIR, MODEL_DIR, RESULTS_DIR, LOGS_DIR]
    
    print("\nVerifying directory structure...")
    for directory in required_dirs:
        if directory.exists():
            print(f"✓ {directory.name}: OK")
        else:
            print(f"✗ {directory.name}: Creating...")
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ {directory.name}: Created")
    
    print("\nDirectory structure verified!")


def get_config_dict():
    """Return configuration as dictionary"""
    config = {
        'project': {
            'name': MODEL_NAME,
            'version': MODEL_VERSION,
            'description': MODEL_DESCRIPTION
        },
        'paths': {
            'base_dir': str(BASE_DIR),
            'data_dir': str(DATA_DIR),
            'model_dir': str(MODEL_DIR),
            'results_dir': str(RESULTS_DIR)
        },
        'dataset': {
            'image_size': IMG_SIZE,
            'num_classes': NUM_CLASSES,
            'class_names': CLASS_NAMES,
            'batch_size': BATCH_SIZE,
            'train_split': TRAIN_SPLIT,
            'val_split': VAL_SPLIT,
            'test_split': TEST_SPLIT
        },
        'model': {
            'base_model': BASE_MODEL,
            'freeze_base': FREEZE_BASE_LAYERS,
            'fine_tune_at': FINE_TUNE_AT_LAYER,
            'dense_units_1': DENSE_UNITS_1,
            'dense_units_2': DENSE_UNITS_2,
            'dropout_rate': DROPOUT_RATE
        },
        'training': {
            'initial_epochs': INITIAL_EPOCHS,
            'fine_tune_epochs': FINE_TUNE_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'fine_tune_lr': FINE_TUNE_LR,
            'optimizer': OPTIMIZER,
            'loss': LOSS_FUNCTION
        },
        'augmentation': {
            'rotation_range': ROTATION_RANGE,
            'width_shift_range': WIDTH_SHIFT_RANGE,
            'height_shift_range': HEIGHT_SHIFT_RANGE,
            'zoom_range': ZOOM_RANGE,
            'horizontal_flip': HORIZONTAL_FLIP,
            'brightness_range': BRIGHTNESS_RANGE
        },
        'system': {
            'random_seed': RANDOM_SEED,
            'use_gpu': USE_GPU,
            'num_workers': NUM_WORKERS
        }
    }
    return config


# ============================================================================
# MAIN (FOR TESTING)
# ============================================================================

if __name__ == "__main__":
    print_config()
    verify_paths()
    
    # Test configuration dictionary
    config_dict = get_config_dict()
    print("\n✓ Configuration dictionary created successfully!")
    print(f"✓ Total configuration items: {sum(len(v) for v in config_dict.values())}")