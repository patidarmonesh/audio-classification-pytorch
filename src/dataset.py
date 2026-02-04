"""Audio dataset implementation with preprocessing and augmentation."""

import os
import logging
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioDataset(Dataset):
    """Dataset class for audio classification with mel-spectrogram features."""
    
    def __init__(self, annotations_file, audio_dir, sample_rate=32000, 
                 duration=4, is_train=True, mean=None, std=None):
        """
        Args:
            annotations_file: Path to CSV with filenames and labels
            audio_dir: Directory containing audio files
            sample_rate: Target sample rate (Hz)
            duration: Target audio duration (seconds)
            is_train: Whether this is training dataset
            mean: Pre-computed mean for normalization
            std: Pre-computed standard deviation for normalization
        """
        # Load annotations
        try:
            self.annotations = pd.read_csv(annotations_file)
        except UnicodeDecodeError:
            self.annotations = pd.read_csv(annotations_file, encoding='latin1')
        
        logger.info(f"Loaded CSV with columns: {list(self.annotations.columns)}")
        
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.target_length = sample_rate * duration
        self.is_train = is_train
        
        # Detect column names
        self.file_column = self._detect_column(['Filename', 'filename', 'file_name'])
        self.label_column = self._detect_column(['Class ID', 'Class_id', 'class_id', 'label'])
        
        # Audio transforms
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        
        # Normalization
        if is_train:
            self.mean, self.std = self._calculate_normalization_stats()
        else:
            if mean is None or std is None:
                raise ValueError("Test dataset requires mean and std from training.")
            self.mean, self.std = mean, std
    
    def _detect_column(self, possible_names):
        """Detect column name from list of possibilities."""
        for name in possible_names:
            if name in self.annotations.columns:
                return name
        raise ValueError(f"Could not find column. Tried: {possible_names}")
    
    def _calculate_normalization_stats(self, num_samples=500):
        """Calculate mean and std from subset of data."""
        num_samples = min(num_samples, len(self))
        indices = np.random.choice(len(self), num_samples, replace=False)
        specs = []
        
        logger.info(f"Calculating normalization stats from {num_samples} samples...")
        
        for idx in indices:
            waveform = self._load_and_preprocess_audio(idx)
            spec = self.amplitude_to_db(self.mel_spectrogram(waveform))
            specs.append(spec)
        
        stacked = torch.stack(specs)
        mean = stacked.mean().item()
        std = stacked.std().item()
        
        logger.info(f"Normalization stats: mean={mean:.4f}, std={std:.4f}")
        return mean, std
    
    def _load_and_preprocess_audio(self, idx):
        """Load and preprocess audio file to fixed length."""
        file_name = str(self.annotations.loc[idx, self.file_column]).strip()
        audio_path = os.path.join(self.audio_dir, file_name)
        
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")
            return torch.zeros(1, self.target_length)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Pad or truncate
        if waveform.shape[1] < self.target_length:
            pad_length = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        else:
            waveform = waveform[:, :self.target_length]
        
        return waveform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        waveform = self._load_and_preprocess_audio(idx)
        spec = self.amplitude_to_db(self.mel_spectrogram(waveform))
        spec = (spec - self.mean) / (self.std + 1e-6)
        
        try:
            label = int(self.annotations.loc[idx, self.label_column])
        except Exception as e:
            logger.error(f"Label conversion error at {idx}: {e}")
            label = -1
        
        return spec, label


class AugmentedDataset(Dataset):
    """Wrapper dataset with data augmentation."""
    
    def __init__(self, base_dataset, freq_mask=15, time_mask=30, noise_std=0.001):
        self.base_dataset = base_dataset
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        self.noise_std = noise_std
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        import random
        
        spec, label = self.base_dataset[idx]
        
        # Time shift
        max_shift = spec.size(-1) // 10
        shift = random.randint(-max_shift, max_shift)
        spec = torch.roll(spec, shifts=shift, dims=-1)
        
        # Amplitude scaling
        spec = spec * random.uniform(0.8, 1.2)
        
        # SpecAugment
        spec = self.freq_mask(spec)
        spec = self.time_mask(spec)
        
        # Add noise
        spec = spec + torch.randn_like(spec) * self.noise_std
        
        return spec, label
