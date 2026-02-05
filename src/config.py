import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw' / 'asl_alphabet_train'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

MODEL_DIR = BASE_DIR / 'models' / 'saved_models'

RESULTS_DIR = BASE_DIR / 'results'

LOGS_DIR = BASE_DIR / 'logs'

for directory in [DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
IMG_CHANNELS = 3

NUM_CLASSES = 29

CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-6, \
    "Data splits must sum to 1.0"

BATCH_SIZE = 32

BASE_MODEL = 'MobileNetV2'

FREEZE_BASE_LAYERS = True
FINE_TUNE_AT_LAYER = 100

DENSE_UNITS_1 = 512
DENSE_UNITS_2 = 256
DROPOUT_RATE = 0.5
DROPOUT_RATE_2 = 0.25

INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 5
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

LEARNING_RATE = 0.001
FINE_TUNE_LR = 0.0001

OPTIMIZER = 'adam'
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-7

LOSS_FUNCTION = 'categorical_crossentropy'

METRICS = ['accuracy']

CHECKPOINT_MONITOR = 'val_accuracy'
CHECKPOINT_MODE = 'max'
SAVE_BEST_ONLY = True

EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MODE = 'min'
RESTORE_BEST_WEIGHTS = True

REDUCE_LR_MONITOR = 'val_loss'
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_MIN_LR = 1e-7
REDUCE_LR_MODE = 'min'

TENSORBOARD_HISTOGRAM_FREQ = 1
TENSORBOARD_WRITE_GRAPH = True
TENSORBOARD_WRITE_IMAGES = False

ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
SHEAR_RANGE = 0.2
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True
VERTICAL_FLIP = False
BRIGHTNESS_RANGE = [0.8, 1.2]
FILL_MODE = 'nearest'

RESCALE = 1. / 255

CONFIDENCE_THRESHOLD = 0.7
TOP_K_PREDICTIONS = 3

WEBCAM_ID = 0
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
WEBCAM_FPS = 30

ROI_SIZE = 300
ROI_COLOR = (0, 255, 0)
ROI_THICKNESS = 3

FONT_FACE = 1
FONT_SCALE = 0.9
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
PREDICTION_COLOR = (0, 255, 0)

RANDOM_SEED = 42

import random
import numpy as np
import tensorflow as tf

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

KAGGLE_DATASET = 'grassknoted/asl-alphabet'
KAGGLE_DATASET_URL = 'https://www.kaggle.com/datasets/grassknoted/asl-alphabet'

TOTAL_IMAGES = 87000
IMAGES_PER_CLASS = 3000

MODEL_NAME = 'ASL_Alphabet_Recognition'
MODEL_VERSION = '1.0.0'
MODEL_DESCRIPTION = 'Real-time ASL Alphabet Recognition using Transfer Learning'

IMAGENET_WEIGHTS = 'imagenet'

USE_GPU = False
if not USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

NUM_WORKERS = 4

USE_MULTIPROCESSING = False
MAX_QUEUE_SIZE = 10

LOG_LEVEL = 'INFO'

VERBOSE_TRAINING = 1
VERBOSE_PREDICTION = 0

PLOT_DPI = 300
PLOT_STYLE = 'seaborn'
FIGURE_SIZE = (12, 8)

CM_FIGSIZE = (16, 14)
CM_CMAP = 'Blues'
CM_ANNOT = True

BEST_MODEL_PATH = MODEL_DIR / 'best_model.h5'
FINAL_MODEL_PATH = MODEL_DIR / 'final_model.h5'
MODEL_ARCHITECTURE_PATH = BASE_DIR / 'models' / 'model_architecture.png'

TRAINING_HISTORY_PLOT = RESULTS_DIR / 'training_history.png'
TRAINING_HISTORY_JSON = RESULTS_DIR / 'training_history.json'
CONFUSION_MATRIX_PLOT = RESULTS_DIR / 'confusion_matrix.png'
CONFUSION_MATRIX_NORMALIZED = RESULTS_DIR / 'confusion_matrix_normalized.png'
CLASSIFICATION_REPORT_FILE = RESULTS_DIR / 'classification_report.txt'
MISCLASSIFICATION_ANALYSIS = RESULTS_DIR / 'misclassification_analysis.txt'
CLASS_DISTRIBUTION_PLOT = RESULTS_DIR / 'class_distribution.png'
SAMPLE_IMAGES_PLOT = RESULTS_DIR / 'sample_images.png'

DATASET_INFO_JSON = DATA_DIR / 'dataset_info.json'

def print_config():
    print("=" * 70)
    print("ASL Alphabet Recognition - Configuration")
    print("=" * 70)
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
    print("=" * 70)

def verify_paths():
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

if __name__ == "__main__":
    print_config()
    verify_paths()

    config_dict = get_config_dict()
    print("\n✓ Configuration dictionary created successfully!")
    print(f"✓ Total configuration items: {sum(len(v) for v in config_dict.values())}")
