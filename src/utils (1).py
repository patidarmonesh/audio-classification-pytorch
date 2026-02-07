"""
Utility functions for audio classification project
"""

import torch
import numpy as np
import random
import logging
import os


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


def setup_logging(log_file='training.log', level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        filename=log_file,
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info("Logging initialized")


def save_checkpoint(model, optimizer, epoch, f1_score, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'f1_score': f1_score
    }
    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    logging.info(f"Checkpoint loaded: {filepath}")
    return checkpoint['epoch'], checkpoint['f1_score']


def count_parameters(model):
    """Count total trainable parameters"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total:,}")
    return total
