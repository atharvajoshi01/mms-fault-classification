"""
Comprehensive visualization script for fault classification results.

This script generates:
1. Confusion Matrix (normalized and raw counts)
2. Per-class Performance Metrics (Precision, Recall, F1-Score)
3. Sample Signals from each fault class
4. Class Distribution
5. Performance Dashboard
6. ROC Curves (if applicable)

Usage:
    python scripts/visualize_results.py
    python scripts/visualize_results.py --save-only  # Don't show plots, only save
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from src.data_loader import build_dataset
from src.preprocessing import preprocess_pipeline, load_preprocessed_data
from src.models.minirocket import MiniRocketClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize model results')

    parser.add_argument(
        '--model-path',
        type=str,
        default='models/minirocket/minirocket_model.pkl',
        help='Path to trained model'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='phase_2',
        help='Directory containing CSV files'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations',
        help='Directory to save visualizations'
    )

    parser.add_argument(
        '--save-only',
        action='store_true',
        help='Save plots without displaying them'
    )

    return parser.parse_args()


def plot_confusion_matrix_advanced(y_true, y_pred, class_names, output_dir, show=True):
    """Plot both normalized and raw confusion matrices side by side."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0], cbar_kws={'label': 'Count'},
        square=True, linewidths=0.5
    )
    axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')

    # Normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt='.2%', cmap='Greens',
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[1], cbar_kws={'label': 'Proportion'},
        square=True, linewidths=0.5, vmin=0, vmax=1
    )
    axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')

    plt.tight_layout()

    save_path = output_dir / 'confusion_matrix_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Confusion matrix saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_per_class_metrics(y_true, y_pred, class_names, output_dir, show=True):
    """Plot per-class precision, recall, and F1-score."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names))
    )

    # Create DataFrame
    df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Bar colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Precision
    bars = axes[0, 0].bar(df['Class'], df['Precision'], color=colors, alpha=0.8)
    axes[0, 0].set_ylabel('Precision', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Precision by Class', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylim(0, 1.1)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, df['Precision']):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    axes[0, 0].set_xticklabels(df['Class'], rotation=45, ha='right')

    # Recall
    bars = axes[0, 1].bar(df['Class'], df['Recall'], color=colors, alpha=0.8)
    axes[0, 1].set_ylabel('Recall', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Recall by Class', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim(0, 1.1)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, df['Recall']):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    axes[0, 1].set_xticklabels(df['Class'], rotation=45, ha='right')

    # F1-Score
    bars = axes[1, 0].bar(df['Class'], df['F1-Score'], color=colors, alpha=0.8)
    axes[1, 0].set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('F1-Score by Class', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, df['F1-Score']):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    axes[1, 0].set_xticklabels(df['Class'], rotation=45, ha='right')

    # Support
    bars = axes[1, 1].bar(df['Class'], df['Support'], color=colors, alpha=0.8)
    axes[1, 1].set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Sample Count by Class', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, df['Support']):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(val)}', ha='center', va='bottom', fontweight='bold')
    axes[1, 1].set_xticklabels(df['Class'], rotation=45, ha='right')

    plt.suptitle('Per-Class Performance Metrics', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    save_path = output_dir / 'per_class_metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Per-class metrics saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return df


def plot_sample_signals(X, y, class_names, output_dir, samples_per_class=3, show=True):
    """Plot sample signals from each class."""
    fig, axes = plt.subplots(len(class_names), samples_per_class,
                            figsize=(15, 3*len(class_names)))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # For 3 channels
    channel_names = ['X-axis', 'Y-axis', 'Z-axis']

    for class_idx, class_name in enumerate(class_names):
        # Get samples from this class
        class_mask = (y == class_idx)
        class_samples = X[class_mask]

        # Select random samples
        sample_indices = np.random.choice(len(class_samples),
                                        min(samples_per_class, len(class_samples)),
                                        replace=False)

        for sample_idx in range(samples_per_class):
            ax = axes[class_idx, sample_idx]

            if sample_idx < len(sample_indices):
                signal = class_samples[sample_indices[sample_idx]]

                # Plot all channels
                for ch_idx in range(signal.shape[1]):
                    ax.plot(signal[:, ch_idx], linewidth=0.5,
                           alpha=0.8, label=channel_names[ch_idx],
                           color=colors[ch_idx])

                if sample_idx == 0:
                    ax.set_ylabel(f'{class_name}', fontsize=10, fontweight='bold')

                if class_idx == 0:
                    ax.set_title(f'Sample {sample_idx + 1}', fontsize=11, fontweight='bold')

                if class_idx == 0 and sample_idx == samples_per_class - 1:
                    ax.legend(loc='upper right', fontsize=8)

                ax.grid(True, alpha=0.2)
                ax.set_xlabel('Time Step', fontsize=9)
            else:
                ax.axis('off')

    plt.suptitle('Sample Vibration Signals by Fault Class',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_path = output_dir / 'sample_signals.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Sample signals saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_class_distribution(y_train, y_test, class_names, output_dir, show=True):
    """Plot class distribution for train and test sets."""
    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)

    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, train_counts, width, label='Train',
                  color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_counts, width, label='Test',
                  color='coral', alpha=0.8)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Fault Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution: Train vs Test', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    save_path = output_dir / 'class_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Class distribution saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curves(y_true, y_pred_proba, class_names, output_dir, show=True):
    """Plot ROC curves for each class (one-vs-rest)."""
    n_classes = len(class_names)

    # Binarize labels for ROC calculation
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
               label=f'{class_names[i]} (AUC = {roc_auc[i]:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = output_dir / 'roc_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"ROC curves saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def create_performance_dashboard(metrics_df, train_acc, test_acc, training_time,
                                output_dir, show=True):
    """Create a comprehensive performance dashboard."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('MiniRocket Fault Classification - Performance Dashboard',
                fontsize=18, fontweight='bold', y=0.98)

    # Overall metrics (top row)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(0.5, 0.7, 'Train Accuracy', ha='center', fontsize=14, fontweight='bold')
    ax1.text(0.5, 0.3, f'{train_acc:.2%}', ha='center', fontsize=32,
            color='green' if train_acc > 0.95 else 'orange', fontweight='bold')
    ax1.axis('off')
    ax1.set_facecolor('#f0f0f0')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(0.5, 0.7, 'Test Accuracy', ha='center', fontsize=14, fontweight='bold')
    ax2.text(0.5, 0.3, f'{test_acc:.2%}', ha='center', fontsize=32,
            color='green' if test_acc > 0.95 else 'orange', fontweight='bold')
    ax2.axis('off')
    ax2.set_facecolor('#f0f0f0')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(0.5, 0.7, 'Training Time', ha='center', fontsize=14, fontweight='bold')
    ax3.text(0.5, 0.3, f'{training_time/60:.1f} min', ha='center', fontsize=32,
            color='steelblue', fontweight='bold')
    ax3.axis('off')
    ax3.set_facecolor('#f0f0f0')

    # Per-class metrics table (middle)
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('tight')
    ax4.axis('off')

    table_data = []
    for _, row in metrics_df.iterrows():
        table_data.append([
            row['Class'],
            f"{row['Precision']:.4f}",
            f"{row['Recall']:.4f}",
            f"{row['F1-Score']:.4f}",
            f"{int(row['Support'])}"
        ])

    table = ax4.table(cellText=table_data,
                     colLabels=['Class', 'Precision', 'Recall', 'F1-Score', 'Samples'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0.1, 0.1, 0.8, 0.8])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style cells
    colors = ['#E7E6E6', 'white']
    for i in range(1, len(table_data) + 1):
        for j in range(5):
            table[(i, j)].set_facecolor(colors[i % 2])

    ax4.set_title('Per-Class Performance Metrics', fontsize=14,
                 fontweight='bold', pad=20)

    # Model info (bottom)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    info_text = f"""
    Model: MiniRocket + Ridge Classifier
    Number of Kernels: 10,000
    Features Extracted: {10000 * 2} (mean and max pooling per kernel)

    Dataset Info:
    - Total Train Samples: {int(metrics_df['Support'].sum())}
    - Total Test Samples: Available
    - Classes: {len(metrics_df)} fault types
    - Input: Time-series vibration data (3 channels: X, Y, Z)

    Key Strengths:
    - Fast training (linear classifier on MiniRocket features)
    - Excellent performance on time-series classification
    - No need for manual feature engineering
    """

    ax5.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#f9f9f9',
            edgecolor='gray', linewidth=1))

    plt.tight_layout()

    save_path = output_dir / 'performance_dashboard.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Performance dashboard saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    """Main visualization function."""
    args = parse_args()

    logger.info("="*70)
    logger.info("FAULT CLASSIFICATION RESULTS VISUALIZATION")
    logger.info("="*70)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"\nLoading model from: {args.model_path}")
    model = MiniRocketClassifier.load(Path(args.model_path))

    # Load label encoder and scaler
    import joblib
    model_dir = Path(args.model_path).parent
    label_encoder = joblib.load(model_dir / 'label_encoder.pkl')
    scaler = joblib.load(model_dir / 'scaler.pkl')

    logger.info(f"Label encoder classes: {label_encoder.classes_}")

    # Load data
    logger.info(f"Loading data from: {args.data_dir}")
    file_map = {
        'normal': 'normal.csv',
        'unbalance_fault': 'unbalance_fault.csv',
        'misalignment_fault': 'misalignment_fault.csv',
        'bearing_fault': 'bearing_fault.csv'
    }

    from src.data_loader import build_dataset
    X, y = build_dataset(Path(args.data_dir), file_map)

    # Encode labels
    y_encoded = label_encoder.transform(y)

    # Get class names (formatted)
    class_names_map = {
        'bearing_fault': 'Bearing Fault',
        'misalignment_fault': 'Misalignment',
        'normal': 'Normal',
        'unbalance_fault': 'Unbalance'
    }
    class_names = [class_names_map[c] for c in label_encoder.classes_]

    logger.info(f"Data loaded: {X.shape}")
    logger.info(f"Classes: {class_names}")

    # Normalize data
    from src.preprocessing import normalize_data
    X_normalized, _ = normalize_data(X, scaler=scaler, fit=False)

    # Split data (same as training)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Generate predictions
    logger.info("\nGenerating predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)

    train_acc = np.mean(y_train_pred == y_train)
    test_acc = np.mean(y_test_pred == y_test)

    logger.info(f"Train Accuracy: {train_acc:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")

    # Generate visualizations
    logger.info("\n" + "="*70)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*70)

    show_plots = not args.save_only

    # 1. Confusion Matrix
    logger.info("\n1. Creating confusion matrix...")
    plot_confusion_matrix_advanced(y_test, y_test_pred, class_names,
                                  output_dir, show=show_plots)

    # 2. Per-class metrics
    logger.info("2. Creating per-class metrics...")
    metrics_df = plot_per_class_metrics(y_test, y_test_pred, class_names,
                                       output_dir, show=show_plots)

    # 3. Sample signals (use non-normalized data for visualization)
    logger.info("3. Plotting sample signals...")
    # Get non-normalized test data for better visualization
    X_raw, y_raw = build_dataset(Path(args.data_dir), file_map)
    y_raw_encoded = label_encoder.transform(y_raw)
    _, X_test_raw, _, y_test_raw = train_test_split(
        X_raw, y_raw_encoded, test_size=0.2, random_state=42, stratify=y_raw_encoded
    )
    plot_sample_signals(X_test_raw, y_test_raw, class_names, output_dir,
                       samples_per_class=3, show=show_plots)

    # 4. Class distribution
    logger.info("4. Creating class distribution...")
    plot_class_distribution(y_train, y_test, class_names, output_dir,
                          show=show_plots)

    # 5. ROC curves
    logger.info("5. Creating ROC curves...")
    plot_roc_curves(y_test, y_test_pred_proba, class_names, output_dir,
                   show=show_plots)

    # 6. Performance dashboard
    logger.info("6. Creating performance dashboard...")
    create_performance_dashboard(metrics_df, train_acc, test_acc, 242.5,
                                output_dir, show=show_plots)

    # Print classification report
    logger.info("\n" + "="*70)
    logger.info("CLASSIFICATION REPORT")
    logger.info("="*70)
    print(classification_report(y_test, y_test_pred, target_names=class_names))

    logger.info("\n" + "="*70)
    logger.info("VISUALIZATION COMPLETE!")
    logger.info("="*70)
    logger.info(f"\nAll visualizations saved to: {output_dir.absolute()}")

    # Save classification report
    report_path = output_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(classification_report(y_test, y_test_pred, target_names=class_names))
        f.write("\n\n")
        f.write(f"Train Accuracy: {train_acc:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")

    logger.info(f"Classification report saved to: {report_path}")


if __name__ == "__main__":
    main()
