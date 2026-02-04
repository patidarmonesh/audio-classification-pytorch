"""Training script for audio classification with cross-validation."""

import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import custom modules
from dataset import AudioDataset, AugmentedDataset
from models import AudioCNN
from utils import set_seed, calculate_metrics, compute_class_weights

# Try importing PyTorch XLA for TPU
try:
    import torch_xla.core.xla_model as xm
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, criterion, optimizer, device, use_tpu=False):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        if use_tpu:
            xm.optimizer_step(optimizer)
            xm.mark_step()
        else:
            optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, return_preds=False, use_tpu=False):
    """Evaluate model on validation/test set."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if use_tpu:
                xm.mark_step()
    
    avg_loss = running_loss / len(loader.dataset)
    f1, prec, rec = calculate_metrics(all_labels, all_preds)
    
    if return_preds:
        return avg_loss, f1, prec, rec, all_labels, all_preds
    return avg_loss, f1, prec, rec


def main(args):
    """Main training function."""
    set_seed(args.seed)
    
    # Device setup
    if TPU_AVAILABLE and args.use_tpu:
        device = xm.xla_device()
        logger.info(f"Using TPU device: {device}")
        use_tpu = True
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        use_tpu = False
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = AudioDataset(
        args.train_annotations,
        args.train_audio_dir,
        sample_rate=args.sample_rate,
        duration=args.duration,
        is_train=True
    )
    
    test_dataset = AudioDataset(
        args.test_annotations,
        args.test_audio_dir,
        sample_rate=args.sample_rate,
        duration=args.duration,
        is_train=False,
        mean=train_dataset.mean,
        std=train_dataset.std
    )
    
    # Compute class weights
    train_labels = train_dataset.annotations[train_dataset.label_column].astype(int).values
    class_weights = compute_class_weights(train_labels, args.num_classes)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    logger.info(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Cross-validation
    if args.num_folds > 1:
        logger.info(f"Starting {args.num_folds}-fold cross-validation...")
        skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(train_labels)), train_labels)):
            logger.info(f"\n{'='*50}")
            logger.info(f"Fold {fold+1}/{args.num_folds}")
            logger.info(f"{'='*50}")
            
            # Create data loaders
            train_loader = DataLoader(
                AugmentedDataset(Subset(train_dataset, train_idx)),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers
            )
            
            val_loader = DataLoader(
                Subset(train_dataset, val_idx),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers
            )
            
            # Initialize model
            model = AudioCNN(num_classes=args.num_classes).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.patience, factor=0.5)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            
            # Training loop
            best_f1 = 0.0
            for epoch in range(args.num_epochs):
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, use_tpu)
                val_loss, f1, prec, rec = evaluate(model, val_loader, criterion, device, use_tpu=use_tpu)
                scheduler.step(val_loss)
                
                logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
                logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                logger.info(f"F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
                
                # Save best model
                if f1 > best_f1:
                    best_f1 = f1
                    model_path = os.path.join(args.output_dir, 'models', f'best_fold{fold+1}.pth')
                    torch.save(model.state_dict(), model_path)
                    logger.info(f"Saved best model: {model_path}")
            
            # Evaluate best model
            model.load_state_dict(torch.load(os.path.join(args.output_dir, 'models', f'best_fold{fold+1}.pth')))
            _, f1, prec, rec = evaluate(model, val_loader, criterion, device, use_tpu=use_tpu)
            fold_metrics.append((f1, prec, rec))
            logger.info(f"Fold {fold+1} Best - F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
        
        # Plot CV results
        metrics = np.array(fold_metrics)
        plt.figure(figsize=(10, 6))
        plt.bar([f'Fold {i+1}' for i in range(args.num_folds)], metrics[:, 0])
        plt.ylabel('F1 Score (macro)')
        plt.title('Cross-Validation F1 Scores')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'plots', 'cv_results.png'))
        plt.close()
        
        avg_f1, avg_prec, avg_rec = metrics.mean(axis=0)
        logger.info(f"\nCV Average - F1: {avg_f1:.4f}, Precision: {avg_prec:.4f}, Recall: {avg_rec:.4f}")
    
    # Final training on full dataset
    logger.info("\nTraining final model on full training data...")
    full_train_loader = DataLoader(
        AugmentedDataset(train_dataset),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    final_model = AudioCNN(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(final_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.patience, factor=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    for epoch in range(args.num_epochs):
        train_loss = train_one_epoch(final_model, full_train_loader, criterion, optimizer, device, use_tpu)
        scheduler.step(train_loss)
        logger.info(f"Final Model Epoch {epoch+1}/{args.num_epochs} | Train Loss: {train_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'models', 'final_model.pth')
    torch.save(final_model.state_dict(), final_model_path)
    logger.info(f"Saved final model: {final_model_path}")
    
    # Test evaluation
    logger.info("\nEvaluating on test set...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    test_loss, test_f1, test_prec, test_rec, true_labels, pred_labels = evaluate(
        final_model, test_loader, criterion, device, return_preds=True, use_tpu=use_tpu
    )
    
    logger.info(f"\nTest Results:")
    logger.info(f"Loss: {test_loss:.4f} | F1: {test_f1:.4f} | Precision: {test_prec:.4f} | Recall: {test_rec:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap='Blues')
    plt.title('Test Set Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'plots', 'confusion_matrix.png'))
    plt.close()
    
    logger.info(f"All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio Classification Training')
    
    # Data arguments
    parser.add_argument('--train_annotations', type=str, required=True, help='Path to train CSV')
    parser.add_argument('--test_annotations', type=str, required=True, help='Path to test CSV')
    parser.add_argument('--train_audio_dir', type=str, required=True, help='Path to train audio directory')
    parser.add_argument('--test_audio_dir', type=str, required=True, help='Path to test audio directory')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=7, help='Number of classes')
    parser.add_argument('--sample_rate', type=int, default=32000, help='Audio sample rate')
    parser.add_argument('--duration', type=int, default=4, help='Audio duration in seconds')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=3, help='Scheduler patience')
    parser.add_argument('--num_folds', type=int, default=3, help='Number of CV folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--use_tpu', action='store_true', help='Use TPU if available')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    
    args = parser.parse_args()
    main(args)
