"""
Model evaluation module for ASL Alphabet Recognition
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from src.config import *
from src.data_preprocessing import ASLDataPreprocessor
from src.model import ASLModel


class ASLEvaluator:
    """Evaluates trained model performance"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to saved model
        """
        if model_path is None:
            model_path = MODEL_DIR / 'best_model.h5'
        
        print(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(str(model_path))
        print("✅ Model loaded successfully!")
        
    def evaluate_on_test_set(self, test_generator):
        """
        Evaluate model on test set
        
        Args:
            test_generator: Test data generator
        """
        print("\n" + "="*70)
        print("Evaluating Model on Test Set")
        print("="*70)
        
        # Get predictions
        test_generator.reset()
        predictions = self.model.predict(test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        
        # Calculate metrics
        test_loss, test_accuracy, test_top3_acc = self.model.evaluate(
            test_generator,
            verbose=1
        )
        
        print("\n" + "="*70)
        print("Test Set Results")
        print("="*70)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Top-3 Accuracy: {test_top3_acc:.4f} ({test_top3_acc*100:.2f}%)")
        
        # Generate classification report
        self._generate_classification_report(y_true, y_pred, test_generator.class_indices)
        
        # Generate confusion matrix
        self._plot_confusion_matrix(y_true, y_pred, test_generator.class_indices)
        
        # Analyze misclassifications
        self._analyze_misclassifications(y_true, y_pred, predictions, test_generator.class_indices)
        
        return test_accuracy, y_true, y_pred
    
    def _generate_classification_report(self, y_true, y_pred, class_indices):
        """Generate and save classification report"""
        # Get class names in correct order
        class_names_ordered = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
        
        # Generate report
        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names_ordered,
            digits=4
        )
        
        print("\n" + "="*70)
        print("Classification Report")
        print("="*70)
        print(report)
        
        # Save to file
        with open(RESULTS_DIR / 'classification_report.txt', 'w') as f:
            f.write("ASL Alphabet Recognition - Classification Report\n")
            f.write("="*70 + "\n\n")
            f.write(report)
        
        print(f"\n✅ Classification report saved to: {RESULTS_DIR / 'classification_report.txt'}")
    
    def _plot_confusion_matrix(self, y_true, y_pred, class_indices):
        """Plot and save confusion matrix"""
        # Get class names in correct order
        class_names_ordered = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(16, 14))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names_ordered,
            yticklabels=class_names_ordered,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - ASL Alphabet Recognition', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to: {RESULTS_DIR / 'confusion_matrix.png'}")
        plt.close()
        
        # Calculate and plot normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(16, 14))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            xticklabels=class_names_ordered,
            yticklabels=class_names_ordered,
            cbar_kws={'label': 'Proportion'},
            vmin=0,
            vmax=1
        )
        plt.title('Normalized Confusion Matrix - ASL Alphabet Recognition', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
        print(f"✅ Normalized confusion matrix saved to: {RESULTS_DIR / 'confusion_matrix_normalized.png'}")
        plt.close()
    
    def _analyze_misclassifications(self, y_true, y_pred, predictions, class_indices):
        """Analyze and report misclassifications"""
        # Get class names
        idx_to_class = {v: k for k, v in class_indices.items()}
        
        # Find misclassifications
        misclassified_indices = np.where(y_true != y_pred)[0]
        
        print("\n" + "="*70)
        print("Misclassification Analysis")
        print("="*70)
        print(f"\nTotal Misclassifications: {len(misclassified_indices)}")
        print(f"Error Rate: {len(misclassified_indices) / len(y_true) * 100:.2f}%")
        
        # Find most confused pairs
        confusion_pairs = {}
        for idx in misclassified_indices:
            true_label = idx_to_class[y_true[idx]]
            pred_label = idx_to_class[y_pred[idx]]
            pair = (true_label, pred_label)
            
            if pair not in confusion_pairs:
                confusion_pairs[pair] = 0
            confusion_pairs[pair] += 1
        
        # Sort by frequency
        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 Most Confused Class Pairs:")
        print("-" * 70)
        for i, ((true_cls, pred_cls), count) in enumerate(sorted_pairs[:10], 1):
            print(f"{i:2d}. True: {true_cls:10s} → Predicted: {pred_cls:10s} | Count: {count:3d}")
        
        # Save detailed misclassification analysis
        with open(RESULTS_DIR / 'misclassification_analysis.txt', 'w', encoding='utf-8') as f:
            f.write("Misclassification Analysis\n")
            f.write("="*70 + "\n\n")
            f.write(f"Total Misclassifications: {len(misclassified_indices)}\n")
            f.write(f"Error Rate: {len(misclassified_indices) / len(y_true) * 100:.2f}%\n\n")
            f.write("All Confused Class Pairs (sorted by frequency):\n")
            f.write("-"*70 + "\n")
            for (true_cls, pred_cls), count in sorted_pairs:
                f.write(f"True: {true_cls:10s} -> Predicted: {pred_cls:10s} | Count: {count:3d}\n")
        
        print(f"\n✅ Detailed analysis saved to: {RESULTS_DIR / 'misclassification_analysis.txt'}")


def main():
    """Main evaluation pipeline"""
    # Create data generators
    preprocessor = ASLDataPreprocessor()
    _, _, test_gen = preprocessor.create_data_generators()
    
    # Evaluate model
    evaluator = ASLEvaluator()
    test_accuracy, y_true, y_pred = evaluator.evaluate_on_test_set(test_gen)
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("\nNext Step: Run real-time inference with:")
    print("  - Streamlit app: streamlit run app/streamlit_app.py")
    print("  - Webcam demo: python app/webcam_demo.py")
    print("="*70)


if __name__ == "__main__":
    main()