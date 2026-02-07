"""
Evaluation utilities and metrics calculation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
from tqdm import tqdm
import logging


def calculate_metrics(y_true, y_pred, average='macro'):
    """Calculate classification metrics"""
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on dataset"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(all_labels, all_preds)

    return avg_loss, metrics, all_labels, all_preds


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')

    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Confusion matrix saved to {save_path}")

    plt.show()
    plt.close()


def print_classification_report(y_true, y_pred, class_names=None):
    """Print detailed classification report"""
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                   zero_division=0)
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    logging.info(f"Classification Report:\n{report}")


def plot_training_history(history, save_path=None):
    """Plot training and validation metrics"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1 Score plot
    axes[1].plot(epochs, history['val_f1'], 'g-', label='Val F1 Score', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Training history plot saved to {save_path}")

    plt.show()
    plt.close()


def plot_fold_comparison(fold_metrics, save_path=None):
    """Plot comparison of fold performances"""
    folds = [f"Fold {i+1}" for i in range(len(fold_metrics))]
    f1_scores = [m['f1'] for m in fold_metrics]
    precisions = [m['precision'] for m in fold_metrics]
    recalls = [m['recall'] for m in fold_metrics]

    x = np.arange(len(folds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width, f1_scores, width, label='F1 Score', color='#2ecc71')
    ax.bar(x, precisions, width, label='Precision', color='#3498db')
    ax.bar(x + width, recalls, width, label='Recall', color='#e74c3c')

    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Cross-Validation Performance by Fold', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Fold comparison plot saved to {save_path}")

    plt.show()
    plt.close()
