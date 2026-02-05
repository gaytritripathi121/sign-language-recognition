"""
Utility functions for ASL Recognition System
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import cv2
from typing import List, Dict, Tuple


def visualize_training_history(metrics_path: str, save_path: str = None):
    """
    Visualize training history from metrics JSON
    
    Args:
        metrics_path: Path to training_metrics.json
        save_path: Optional path to save figure
    """
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(metrics['train_losses'], label='Train Loss', linewidth=2)
    axes[0].plot(metrics['val_losses'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(metrics['train_accs'], label='Train Accuracy', linewidth=2)
    axes[1].plot(metrics['val_accs'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_video_frames(video_path: str, num_frames: int = 8, save_path: str = None):
    """
    Visualize sample frames from a video
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to display
        save_path: Optional path to save figure
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    # Plot frames
    fig, axes = plt.subplots(2, num_frames // 2, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, (frame, idx) in enumerate(zip(frames, frame_indices)):
        axes[i].imshow(frame)
        axes[i].set_title(f'Frame {idx}', fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_class_distribution(data_dir: str, save_path: str = None):
    """
    Plot class distribution from dataset statistics
    
    Args:
        data_dir: Directory containing dataset_stats.json
        save_path: Optional path to save figure
    """
    stats_path = Path(data_dir) / 'dataset_stats.json'
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    label_dist = stats['label_distribution']
    
    # Sort by frequency
    sorted_items = sorted(label_dist.items(), key=lambda x: x[1], reverse=True)
    labels, counts = zip(*sorted_items)
    
    # Plot
    plt.figure(figsize=(15, 6))
    
    if len(labels) > 50:
        # Show top 50 if too many classes
        plt.bar(range(50), counts[:50])
        plt.xlabel('Top 50 Classes (by frequency)', fontsize=12)
    else:
        plt.bar(range(len(labels)), counts)
        plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
        plt.xlabel('Sign Class', fontsize=12)
    
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Class Distribution in Training Set', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def ensemble_predict(models: List[torch.nn.Module], 
                     frames: torch.Tensor, 
                     device: torch.device) -> torch.Tensor:
    """
    Ensemble prediction from multiple models
    
    Args:
        models: List of trained models
        frames: Input frames tensor
        device: torch device
        
    Returns:
        averaged_probs: Ensemble predictions
    """
    all_probs = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            outputs = model(frames.to(device))
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs)
    
    # Average probabilities
    averaged_probs = torch.stack(all_probs).mean(dim=0)
    
    return averaged_probs


def create_video_from_frames(frames: np.ndarray, 
                            output_path: str, 
                            fps: int = 10):
    """
    Create video from numpy frames
    
    Args:
        frames: Array of shape (num_frames, H, W, C)
        output_path: Path to save video
        fps: Frames per second
    """
    height, width = frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()


def get_model_summary(model: torch.nn.Module, input_size: Tuple[int, ...]) -> Dict:
    """
    Get detailed model summary
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, T, C, H, W)
        
    Returns:
        summary: Dictionary with model information
    """
    def count_params(module):
        return sum(p.numel() for p in module.parameters())
    
    def count_trainable_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    summary = {
        'total_params': count_params(model),
        'trainable_params': count_trainable_params(model),
        'input_size': input_size,
        'model_class': model.__class__.__name__
    }
    
    # Layer-wise breakdown
    layers = {}
    for name, module in model.named_children():
        layers[name] = {
            'params': count_params(module),
            'trainable': count_trainable_params(module)
        }
    
    summary['layers'] = layers
    
    return summary


def print_model_summary(model: torch.nn.Module, input_size: Tuple[int, ...]):
    """Print formatted model summary"""
    summary = get_model_summary(model, input_size)
    
    print("=" * 70)
    print(f"Model: {summary['model_class']}")
    print("=" * 70)
    print(f"Input size: {summary['input_size']}")
    print(f"Total parameters: {summary['total_params']:,}")
    print(f"Trainable parameters: {summary['trainable_params']:,}")
    print(f"Non-trainable parameters: {summary['total_params'] - summary['trainable_params']:,}")
    print("\nLayer-wise breakdown:")
    print("-" * 70)
    print(f"{'Layer':<20} {'Parameters':>15} {'Trainable':>15}")
    print("-" * 70)
    
    for name, info in summary['layers'].items():
        print(f"{name:<20} {info['params']:>15,} {info['trainable']:>15,}")
    
    print("=" * 70)


def save_predictions_csv(predictions: np.ndarray, 
                        true_labels: np.ndarray,
                        idx_to_label: Dict,
                        output_path: str):
    """
    Save predictions to CSV file
    
    Args:
        predictions: Predicted class indices
        true_labels: True class indices
        idx_to_label: Mapping from index to label
        output_path: Path to save CSV
    """
    import pandas as pd
    
    df = pd.DataFrame({
        'true_label': [idx_to_label[i] for i in true_labels],
        'predicted_label': [idx_to_label[i] for i in predictions],
        'correct': predictions == true_labels
    })
    
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def calculate_per_class_metrics(confusion_matrix: np.ndarray, 
                                idx_to_label: Dict) -> pd.DataFrame:
    """
    Calculate per-class precision, recall, F1
    
    Args:
        confusion_matrix: Confusion matrix
        idx_to_label: Mapping from index to label
        
    Returns:
        df: DataFrame with per-class metrics
    """
    import pandas as pd
    
    num_classes = len(idx_to_label)
    metrics = []
    
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        tn = confusion_matrix.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.append({
            'class': idx_to_label[i],
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': confusion_matrix[i, :].sum()
        })
    
    df = pd.DataFrame(metrics)
    return df.sort_values('f1_score', ascending=False)


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test model summary
    from model import get_model
    
    model = get_model(num_classes=100, backbone='mobilenet_v2')
    print_model_summary(model, (4, 32, 3, 224, 224))