"""
Training pipeline for ASL Alphabet Recognition
"""
import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from src.config import *
from src.data_preprocessing import ASLDataPreprocessor
from src.model import ASLModel, create_callbacks


class ASLTrainer:
    """Handles model training"""
    
    def __init__(self):
        self.model = None
        self.history = None
        self.fine_tune_history = None
        
    def train(self, train_generator, val_generator):
        """
        Complete training pipeline with initial training and fine-tuning
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
        """
        # Phase 1: Initial Training (frozen base)
        print("\n" + "="*70)
        print("PHASE 1: Initial Training with Frozen Base Model")
        print("="*70)
        
        asl_model = ASLModel(model_name=BASE_MODEL)
        model = asl_model.build_model(freeze_base=True)
        asl_model.compile_model(learning_rate=LEARNING_RATE)
        
        callbacks = create_callbacks()
        
        self.history = model.fit(
            train_generator,
            epochs=INITIAL_EPOCHS,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tuning (unfreeze base)
        print("\n" + "="*70)
        print("PHASE 2: Fine-Tuning with Unfrozen Base Model")
        print("="*70)
        
        asl_model.unfreeze_base_model(fine_tune_at=FINE_TUNE_AT_LAYER)
        asl_model.compile_model(learning_rate=FINE_TUNE_LR)
        
        self.fine_tune_history = model.fit(
            train_generator,
            epochs=FINE_TUNE_EPOCHS,
            validation_data=val_generator,
            initial_epoch=INITIAL_EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        self.model = asl_model
        
        # Save training history
        self._save_training_history()
        
        # Plot training results
        self._plot_training_history()
        
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        
        return self.model
    
    def _save_training_history(self):
        """Save training history to JSON"""
        history_dict = {
            'initial_training': {
                'loss': [float(x) for x in self.history.history['loss']],
                'accuracy': [float(x) for x in self.history.history['accuracy']],
                'val_loss': [float(x) for x in self.history.history['val_loss']],
                'val_accuracy': [float(x) for x in self.history.history['val_accuracy']]
            },
            'fine_tuning': {
                'loss': [float(x) for x in self.fine_tune_history.history['loss']],
                'accuracy': [float(x) for x in self.fine_tune_history.history['accuracy']],
                'val_loss': [float(x) for x in self.fine_tune_history.history['val_loss']],
                'val_accuracy': [float(x) for x in self.fine_tune_history.history['val_accuracy']]
            }
        }
        
        with open(RESULTS_DIR / 'training_history.json', 'w') as f:
            json.dump(history_dict, f, indent=4)
        
        print(f"\nTraining history saved to: {RESULTS_DIR / 'training_history.json'}")
    
    def _plot_training_history(self):
        """Plot training and validation metrics"""
        # Combine histories
        acc = self.history.history['accuracy'] + self.fine_tune_history.history['accuracy']
        val_acc = self.history.history['val_accuracy'] + self.fine_tune_history.history['val_accuracy']
        loss = self.history.history['loss'] + self.fine_tune_history.history['loss']
        val_loss = self.history.history['val_loss'] + self.fine_tune_history.history['val_loss']
        
        epochs_range = range(len(acc))
        
        plt.figure(figsize=(14, 5))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy', linewidth=2)
        plt.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=2)
        plt.axvline(x=INITIAL_EPOCHS, color='red', linestyle='--', label='Fine-tuning Start')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Model Accuracy', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss', linewidth=2)
        plt.plot(epochs_range, val_loss, label='Validation Loss', linewidth=2)
        plt.axvline(x=INITIAL_EPOCHS, color='red', linestyle='--', label='Fine-tuning Start')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Model Loss', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
        print(f"Training plots saved to: {RESULTS_DIR / 'training_history.png'}")
        plt.close()


def main():
    """Main training pipeline"""
    # Set random seeds for reproducibility
    tf.random.set_seed(RANDOM_SEED)
    
    # Create data generators
    preprocessor = ASLDataPreprocessor()
    train_gen, val_gen, test_gen = preprocessor.create_data_generators()
    
    # Train model
    trainer = ASLTrainer()
    model = trainer.train(train_gen, val_gen)
    
    # Print final results
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)
    
    final_train_acc = trainer.fine_tune_history.history['accuracy'][-1]
    final_val_acc = trainer.fine_tune_history.history['val_accuracy'][-1]
    final_train_loss = trainer.fine_tune_history.history['loss'][-1]
    final_val_loss = trainer.fine_tune_history.history['val_loss'][-1]
    
    print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    
    print("\n" + "="*70)
    print("Next Step: Run 'python src/evaluate.py' to evaluate on test set")
    print("="*70)


if __name__ == "__main__":
    main()