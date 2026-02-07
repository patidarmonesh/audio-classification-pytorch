"""
Neural Network Architectures for Audio Classification
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with skip connections"""

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for first convolution
        """
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsample for skip connection if dimensions change
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AudioCNN(nn.Module):
    """Custom CNN architecture for audio classification"""

    def __init__(self, num_classes=7, dropout=0.5):
        """
        Args:
            num_classes: Number of output classes
            dropout: Dropout rate for classifier
        """
        super(AudioCNN, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Residual blocks with increasing channels
        self.resblock1 = ResidualBlock(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.resblock2 = ResidualBlock(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.resblock3 = ResidualBlock(128, 256)

        # Global average pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Input: (batch, 1, 128, time_steps)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.resblock1(x)
        x = self.pool2(x)

        x = self.resblock2(x)
        x = self.pool3(x)

        x = self.resblock3(x)

        # Global pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        # Classification
        x = self.classifier(x)

        return x


def get_model(num_classes=7, dropout=0.5):
    """Factory function to create model"""
    model = AudioCNN(num_classes=num_classes, dropout=dropout)
    return model
