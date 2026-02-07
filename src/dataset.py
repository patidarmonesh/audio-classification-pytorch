"""
Audio Dataset and Preprocessing Module
"""

import os
import torch
import torchaudio
import pandas as pd
import numpy as np
import random
import logging
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """Dataset for loading and preprocessing audio files"""

    def __init__(self, annotations_file, audio_dir, sample_rate=32000, 
                 duration=4, is_train=True, mean=None, std=None):
        """
        Args:
            annotations_file: Path to CSV with audio metadata
            audio_dir: Directory containing audio files
            sample_rate: Target sample rate (Hz)
            duration: Target audio duration (seconds)
            is_train: Whether this is training dataset
            mean: Normalization mean (required for test set)
            std: Normalization std (required for test set)
        """
        # Load annotations
        try:
            self.annotations = pd.read_csv(annotations_file)
        except UnicodeDecodeError:
            self.annotations = pd.read_csv(annotations_file, encoding='latin1')

        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.target_length = sample_rate * duration
        self.is_train = is_train

        # Detect column names
        self.file_column = 'Filename' if 'Filename' in self.annotations.columns else 'filename'
        if 'Class ID' in self.annotations.columns:
            self.label_column = 'Class ID'
        elif 'Class_id' in self.annotations.columns:
            self.label_column = 'Class_id'
        else:
            raise ValueError("Missing required label column")

        # Audio transforms
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

        # Calculate or set normalization stats
        if is_train:
            self._calculate_normalization_stats()
        else:
            if mean is None or std is None:
                raise ValueError("Test dataset requires mean and std from training")
            self.mean, self.std = mean, std

        print(f"Loaded {len(self)} samples from {annotations_file}")
        print(f"Normalization: mean={self.mean:.4f}, std={self.std:.4f}")

    def _calculate_normalization_stats(self):
        """Calculate mean and std for normalization from sample"""
        num_samples = min(500, len(self))
        indices = np.random.choice(len(self), num_samples, replace=False)
        specs = []

        for idx in indices:
            try:
                waveform = self._load_and_preprocess_audio(idx)
                spec = self.amplitude_to_db(self.mel_spectrogram(waveform))
                specs.append(spec)
            except Exception as e:
                logging.warning(f"Skipped sample {idx} during normalization: {e}")

        stacked = torch.stack(specs)
        self.mean = stacked.mean()
        self.std = stacked.std()

    def _load_and_preprocess_audio(self, idx):
        """Load and preprocess single audio file"""
        file_name = str(self.annotations.loc[idx, self.file_column]).strip()
        audio_path = os.path.join(self.audio_dir, file_name)

        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            logging.error(f"Error loading {audio_path}: {e}")
            return torch.zeros(1, self.target_length)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
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
        """Get preprocessed spectrogram and label"""
        waveform = self._load_and_preprocess_audio(idx)

        # Convert to mel-spectrogram
        spec = self.mel_spectrogram(waveform)
        spec = self.amplitude_to_db(spec)

        # Normalize
        spec = (spec - self.mean) / (self.std + 1e-6)

        # Get label
        try:
            label = int(self.annotations.loc[idx, self.label_column])
        except Exception as e:
            logging.error(f"Label error at index {idx}: {e}")
            label = -1

        return spec, label


class AugmentDataset(Dataset):
    """Wrapper dataset for data augmentation"""

    def __init__(self, base_dataset):
        """
        Args:
            base_dataset: Base AudioDataset to augment
        """
        self.base_dataset = base_dataset
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=30)
        self.noise_std = 0.001

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """Apply augmentations to spectrogram"""
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

        # Add Gaussian noise
        spec = spec + torch.randn_like(spec) * self.noise_std

        return spec, label
