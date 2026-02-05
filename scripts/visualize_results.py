"""
Visualize training results and model performance
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

# Load training history
history_file = Path('results/training_history.json')

if not history_file.exists():
    print("Error: Training history not found. Run training first.")
    exit(1)

with open(history_file, 'r') as f:
    history = json.load(f)

# Combine histories
acc = history['initial_training']['accuracy'] + history['fine_tuning']['accuracy']
val_acc = history['initial_training']['val_accuracy'] + history['fine_tuning']['val_accuracy']
loss = history['initial_training']['loss'] + history['fine_tuning']['loss']
val_loss = history['initial_training']['val_loss'] + history['fine_tuning']['val_loss']

epochs = range(1, len(acc) + 1)
fine_tune_start = len(history['initial_training']['accuracy'])

# Create comprehensive plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Accuracy plot
axes[0, 0].plot(epochs, acc, 'b-', label='Training', linewidth=2)
axes[0, 0].plot(epochs, val_acc, 'r-', label='Validation', linewidth=2)
axes[0, 0].axvline(x=fine_tune_start, color='green', linestyle='--', 
                   label='Fine-tuning Start', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Accuracy', fontsize=12)
axes[0, 0].set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss plot
axes[0, 1].plot(epochs, loss, 'b-', label='Training', linewidth=2)
axes[0, 1].plot(epochs, val_loss, 'r-', label='Validation', linewidth=2)
axes[0, 1].axvline(x=fine_tune_start, color='green', linestyle='--', 
                   label='Fine-tuning Start', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Loss', fontsize=12)
axes[0, 1].set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Learning curve
train_sizes = list(range(1, len(acc) + 1))
axes[1, 0].plot(train_sizes, [a * 100 for a in acc], 'b-', label='Training', linewidth=2)
axes[1, 0].plot(train_sizes, [a * 100 for a in val_acc], 'r-', label='Validation', linewidth=2)
axes[1, 0].fill_between(train_sizes, [a * 100 for a in acc], 
                        [a * 100 for a in val_acc], alpha=0.2)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Accuracy (%)', fontsize=12)
axes[1, 0].set_title('Learning Curve', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Performance summary
final_metrics = {
    'Train Acc': acc[-1] * 100,
    'Val Acc': val_acc[-1] * 100,
    'Train Loss': loss[-1],
    'Val Loss': val_loss[-1]
}

metrics_text = "\n".join([f"{k}: {v:.2f}" for k, v in final_metrics.items()])
axes[1, 1].text(0.5, 0.5, metrics_text, 
                ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 1].set_title('Final Metrics', fontsize=14, fontweight='bold')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('results/comprehensive_training_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved to: results/comprehensive_training_analysis.png")
plt.show()