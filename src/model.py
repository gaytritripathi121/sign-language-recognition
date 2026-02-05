"""
Model architecture for ASL Alphabet Recognition
Implements transfer learning with MobileNetV2 or EfficientNetB0
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.optimizers import Adam

from src.config import *


class ASLModel:
    """ASL Alphabet Recognition Model using Transfer Learning"""
    
    def __init__(self, model_name: str = BASE_MODEL):
        """
        Initialize model
        
        Args:
            model_name: 'MobileNetV2' or 'EfficientNetB0'
        """
        self.model_name = model_name
        self.model = None
        self.base_model = None
        
    def build_model(self, freeze_base: bool = FREEZE_BASE_LAYERS) -> models.Model:
        """
        Build the model architecture
        
        Args:
            freeze_base: Whether to freeze base model layers
            
        Returns:
            Compiled Keras model
        """
        print("\n" + "="*60)
        print(f"Building Model: {self.model_name}")
        print("="*60)
        
        # Load pre-trained base model
        if self.model_name == 'MobileNetV2':
            self.base_model = MobileNetV2(
                input_shape=(*IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
        elif self.model_name == 'EfficientNetB0':
            self.base_model = EfficientNetB0(
                input_shape=(*IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # Freeze base model layers
        self.base_model.trainable = not freeze_base
        
        # Build complete model
        inputs = layers.Input(shape=(*IMG_SIZE, 3))
        
        # Base model
        x = self.base_model(inputs, training=False)
        
        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(DROPOUT_RATE)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(DROPOUT_RATE / 2)(x)
        outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        print(f"\nBase Model: {self.model_name}")
        print(f"Base Layers Frozen: {freeze_base}")
        print(f"Total Layers: {len(self.model.layers)}")
        print(f"Trainable Parameters: {self.count_trainable_params():,}")
        
        return self.model
    
    def compile_model(self, learning_rate: float = LEARNING_RATE):
        """
        Compile the model
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        print(f"\nModel compiled with learning rate: {learning_rate}")
    
    def unfreeze_base_model(self, fine_tune_at: int = FINE_TUNE_AT_LAYER):
        """
        Unfreeze base model for fine-tuning
        
        Args:
            fine_tune_at: Layer index to start fine-tuning from
        """
        print("\n" + "="*60)
        print("Unfreezing Base Model for Fine-Tuning")
        print("="*60)
        
        self.base_model.trainable = True
        
        # Freeze layers before fine_tune_at
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        print(f"Fine-tuning from layer {fine_tune_at}")
        print(f"Trainable Parameters: {self.count_trainable_params():,}")
    
    def count_trainable_params(self) -> int:
        """Count trainable parameters"""
        return sum([tf.size(w).numpy() for w in self.model.trainable_weights])
    
    def summary(self):
        """Print model summary"""
        return self.model.summary()
    
    def save_model(self, filepath: str = None):
        """Save model to disk"""
        if filepath is None:
            filepath = MODEL_DIR / 'best_model.h5'
        self.model.save(str(filepath))  # Convert to string
        print(f"\nModel saved to: {filepath}")
    
    def load_model(self, filepath: str = None):
        """Load model from disk"""
        if filepath is None:
            filepath = MODEL_DIR / 'best_model.h5'
        self.model = models.load_model(str(filepath))  # Convert to string
        print(f"\nModel loaded from: {filepath}")
        return self.model
    
    def plot_model_architecture(self, filename: str = 'model_architecture.png'):
        """Plot and save model architecture"""
        filepath = BASE_DIR / 'models' / filename
        tf.keras.utils.plot_model(
            self.model,
            to_file=str(filepath),  # Convert to string
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            dpi=96
        )
        print(f"\nModel architecture saved to: {filepath}")


def create_callbacks():
    """Create training callbacks"""
    callbacks = [
        # Model checkpoint - save best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / 'best_model.h5'),  # Convert Path to string
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=str(BASE_DIR / 'logs'),  # Convert Path to string
            histogram_freq=1
        )
    ]
    
    return callbacks


def main():
    """Test model building"""
    # Build and compile model
    asl_model = ASLModel(model_name=BASE_MODEL)
    model = asl_model.build_model(freeze_base=True)
    asl_model.compile_model()
    
    # Print summary
    print("\n" + "="*60)
    print("Model Summary")
    print("="*60)
    asl_model.summary()
    
    # Plot architecture (optional - requires graphviz)
    try:
        asl_model.plot_model_architecture()
    except Exception as e:
        print(f"Could not plot model architecture: {e}")
        print("To enable plotting, install: pip install pydot graphviz")


if __name__ == "__main__":
    main()