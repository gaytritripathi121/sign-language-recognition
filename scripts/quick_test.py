

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


from src.config import Config
from src.data_preprocessing import ASLDataPreprocessor
from src.model import ASLModel
from src.inference import ImprovedASLPredictor



print("=" * 70)
print("ASL Recognition System - Quick Test")
print("=" * 70)

# Test 1: Config
print("\n✓ Testing configuration...")
print(f"  - Image size: {Config.IMG_SIZE}")
print(f"  - Batch size: {Config.BATCH_SIZE}")
print(f"  - Number of classes: {Config.NUM_CLASSES}")

# Test 2: Data Preprocessor
print("\n✓ Testing data preprocessor...")
preprocessor = ASLDataPreprocessor()
print("  - Data preprocessor initialized")

# Test 3: Model
print("\n✓ Testing model...")
model_builder = ASLModel()
model = model_builder.build_model()
print(f"  - Model created with {model_builder.count_trainable_params():,} parameters")

# Test 4: Predictor
model_path = Config.MODEL_DIR / "best_model.h5"
if model_path.exists():
    print("\n✓ Testing predictor...")
    predictor = ImprovedASLPredictor(model_path)
    print("  - Predictor loaded successfully")
else:
    print("\n⚠ Predictor test skipped - no trained model found")

print("\n" + "=" * 70)
print("All tests completed ✓")
print("=" * 70)
