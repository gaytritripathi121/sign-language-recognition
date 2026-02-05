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
    def __init__(self, model_name: str = BASE_MODEL):
        self.model_name = model_name
        self.model = None
        self.base_model = None

    def build_model(self, freeze_base: bool = FREEZE_BASE_LAYERS) -> models.Model:
        print("\n" + "=" * 60)
        print(f"Building Model: {self.model_name}")
        print("=" * 60)

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

        self.base_model.trainable = not freeze_base

        inputs = layers.Input(shape=(*IMG_SIZE, 3))
        x = self.base_model(inputs, training=False)
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
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
            ]
        )
        print(f"\nModel compiled with learning rate: {learning_rate}")

    def unfreeze_base_model(self, fine_tune_at: int = FINE_TUNE_AT_LAYER):
        print("\n" + "=" * 60)
        print("Unfreezing Base Model for Fine-Tuning")
        print("=" * 60)

        self.base_model.trainable = True

        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False

        print(f"Fine-tuning from layer {fine_tune_at}")
        print(f"Trainable Parameters: {self.count_trainable_params():,}")

    def count_trainable_params(self) -> int:
        return sum(tf.size(w).numpy() for w in self.model.trainable_weights)

    def summary(self):
        return self.model.summary()

    def save_model(self, filepath: str = None):
        if filepath is None:
            filepath = MODEL_DIR / 'best_model.h5'
        self.model.save(str(filepath))
        print(f"\nModel saved to: {filepath}")

    def load_model(self, filepath: str = None):
        if filepath is None:
            filepath = MODEL_DIR / 'best_model.h5'
        self.model = models.load_model(str(filepath))
        print(f"\nModel loaded from: {filepath}")
        return self.model

    def plot_model_architecture(self, filename: str = 'model_architecture.png'):
        filepath = BASE_DIR / 'models' / filename
        tf.keras.utils.plot_model(
            self.model,
            to_file=str(filepath),
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            dpi=96
        )
        print(f"\nModel architecture saved to: {filepath}")


def create_callbacks():
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(BASE_DIR / 'logs'),
            histogram_freq=1
        )
    ]

    return callbacks


def main():
    asl_model = ASLModel(model_name=BASE_MODEL)
    asl_model.build_model(freeze_base=True)
    asl_model.compile_model()

    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)
    asl_model.summary()

    try:
        asl_model.plot_model_architecture()
    except Exception as e:
        print(f"Could not plot model architecture: {e}")
        print("To enable plotting, install: pip install pydot graphviz")


if __name__ == "__main__":
    main()
