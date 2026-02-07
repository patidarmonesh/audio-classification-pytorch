"""
Main Training Script for Audio Classification
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import logging
from tqdm import tqdm

from dataset import AudioDataset, AugmentDataset
from models import get_model
from evaluate import (
    evaluate_model, plot_confusion_matrix, 
    print_classification_report, plot_training_history, plot_fold_comparison
)
from utils import set_seed, setup_logging, count_parameters


def train_one_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training")
    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        progress_bar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(dataloader.dataset)

    return epoch_loss


def train_fold(model, train_loader, val_loader, criterion, optimizer, 
               scheduler, device, num_epochs, fold_num, output_dir):
    """Train model for single fold"""
    best_f1 = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': []
    }

    for epoch in range(num_epochs):
        print(f"\nFold {fold_num} - Epoch {epoch+1}/{num_epochs}")
        print("-" * 60)

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_metrics['f1'])

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f} | "
              f"Precision: {val_metrics['precision']:.4f} | "
              f"Recall: {val_metrics['recall']:.4f}")

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            save_path = os.path.join(output_dir, 'models', f'best_fold{fold_num}.pth')
            torch.save(model.state_dict(), save_path)
            logging.info(f"Best model saved for fold {fold_num} with F1: {best_f1:.4f}")

    return best_f1, history


def main(args):
    # Setup
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)

    log_file = os.path.join(args.output_dir, 'training.log')
    setup_logging(log_file)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    logging.info(f"Using device: {device}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = AudioDataset(
        annotations_file=args.train_annotations,
        audio_dir=args.train_audio_dir,
        sample_rate=args.sample_rate,
        duration=args.duration,
        is_train=True
    )

    test_dataset = AudioDataset(
        annotations_file=args.test_annotations,
        audio_dir=args.test_audio_dir,
        sample_rate=args.sample_rate,
        duration=args.duration,
        is_train=False,
        mean=train_dataset.mean,
        std=train_dataset.std
    )

    # Compute class weights
    train_labels = train_dataset.annotations[train_dataset.label_column].values
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)

    print(f"\nClass distribution: {class_counts}")
    print(f"Class weights: {class_weights.cpu().numpy()}")

    # Cross-validation
    print(f"\nStarting {args.num_folds}-Fold Cross-Validation...")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(train_labels)), train_labels)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{args.num_folds}")
        print(f"{'='*60}")

        # Create data loaders
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(
            AugmentDataset(train_subset),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        # Model, optimizer, criterion
        model = get_model(num_classes=args.num_classes, dropout=0.5).to(device)

        if fold == 0:  # Print architecture once
            print("\nModel Architecture:")
            print(model)
            count_parameters(model)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=args.patience,
            factor=0.5,
            verbose=True
        )

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Train fold
        best_f1, history = train_fold(
            model, train_loader, val_loader, criterion, optimizer,
            scheduler, device, args.num_epochs, fold+1, args.output_dir
        )

        # Load best model and evaluate
        model.load_state_dict(
            torch.load(os.path.join(args.output_dir, 'models', f'best_fold{fold+1}.pth'))
        )
        _, val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)

        fold_metrics.append(val_metrics)

        print(f"\nFold {fold+1} Best Metrics:")
        print(f"F1: {val_metrics['f1']:.4f} | "
              f"Precision: {val_metrics['precision']:.4f} | "
              f"Recall: {val_metrics['recall']:.4f}")

    # Cross-validation summary
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")

    avg_metrics = {
        'f1': np.mean([m['f1'] for m in fold_metrics]),
        'precision': np.mean([m['precision'] for m in fold_metrics]),
        'recall': np.mean([m['recall'] for m in fold_metrics])
    }

    print(f"Average F1: {avg_metrics['f1']:.4f}")
    print(f"Average Precision: {avg_metrics['precision']:.4f}")
    print(f"Average Recall: {avg_metrics['recall']:.4f}")

    # Plot fold comparison
    plot_fold_comparison(
        fold_metrics,
        save_path=os.path.join(args.output_dir, 'plots', 'cv_results.png')
    )

    # Final training on full dataset
    print(f"\n{'='*60}")
    print("FINAL TRAINING ON FULL DATASET")
    print(f"{'='*60}")

    full_train_loader = DataLoader(
        AugmentDataset(train_dataset),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    final_model = get_model(num_classes=args.num_classes, dropout=0.5).to(device)
    optimizer = optim.AdamW(final_model.parameters(), lr=args.lr, 
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     patience=args.patience, factor=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        train_loss = train_one_epoch(final_model, full_train_loader, criterion, 
                                     optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

    # Save final model
    final_model_path = os.path.join(args.output_dir, 'models', 'final_model.pth')
    torch.save(final_model.state_dict(), final_model_path)
    print(f"\nFinal model saved to {final_model_path}")

    # Test evaluation
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION")
    print(f"{'='*60}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loss, test_metrics, y_true, y_pred = evaluate_model(
        final_model, test_loader, criterion, device
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")

    # Class names
    class_names = ['Cough', 'Sneeze', 'Snore', 'Laugh', 'Crying', 'Sniff', 'Throat_Clear']

    # Print classification report
    print_classification_report(y_true, y_pred, class_names=class_names)

    # Plot confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        class_names=class_names,
        save_path=os.path.join(args.output_dir, 'plots', 'confusion_matrix.png')
    )

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Classification Training")

    # Data arguments
    parser.add_argument('--train_annotations', type=str, required=True,
                       help='Path to training annotations CSV')
    parser.add_argument('--test_annotations', type=str, required=True,
                       help='Path to test annotations CSV')
    parser.add_argument('--train_audio_dir', type=str, required=True,
                       help='Path to training audio directory')
    parser.add_argument('--test_audio_dir', type=str, required=True,
                       help='Path to test audio directory')

    # Model arguments
    parser.add_argument('--num_classes', type=int, default=7,
                       help='Number of classes')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=3,
                       help='LR scheduler patience')
    parser.add_argument('--num_folds', type=int, default=3,
                       help='Number of CV folds')

    # Audio arguments
    parser.add_argument('--sample_rate', type=int, default=32000,
                       help='Audio sample rate')
    parser.add_argument('--duration', type=int, default=4,
                       help='Audio duration in seconds')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')

    args = parser.parse_args()

    main(args)
