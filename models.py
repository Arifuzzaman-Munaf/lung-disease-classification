import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F



class PreTrainedModel(nn.Module):
    """
    Base wrapper for pretrained torchvision models.
    Freezes weights optionally and replaces classifier outside this class.
    """
    def __init__(self, model, number_of_classes, freeze_weights = True):
        super().__init__()
        self.model = model
        if freeze_weights:
            # Disable gradient updates for all parameters
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)


class ResNet50(PreTrainedModel):
    """ResNet50 backbone pretrained on ImageNet with a custom classifier."""
    def __init__(self, number_of_classes = 8, freeze_weights = True):
        # Load pretrained weights
        model = torchvision.models.resnet50(weights="IMAGENET1K_V1")

        super().__init__(model, number_of_classes, freeze_weights)

        # Replace the final fully connected layer
        model.fc = nn.Linear(model.fc.in_features, number_of_classes)
        


class MobileNetV3Small(PreTrainedModel):
    """MobileNetV3-Small backbone pretrained on ImageNet with a custom classifier."""
    def __init__(self, number_of_classes= 8, freeze_weights = True):
        # Load pretrained weights
        model = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")

        super().__init__(model, number_of_classes, freeze_weights)
        
        # Replace the final classifier layer
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, number_of_classes)
        


class SkipBlock(nn.Module):
    """
    Residual block with two convolutional layers, batch normalization, and skip connection.
    Keeps spatial dimensions unchanged.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # First convolutional layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

        # Second convolutional layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W)
        
        Returns:
            torch.Tensor: Output feature map after residual addition.
        """
        x1 = self.layer1(x)       # First convolution stage
        x2 = self.layer2(x1)      # Second convolution stage
        return x1 + x2            # Residual skip connection


class CustomResidualCNN(nn.Module):
    """
    Custom CNN model with residual connections and stride-based downsampling.
    Uses global average pooling instead of flattening to reduce parameters and overfitting.
    """
    def __init__(self, number_of_classes: int = 8):
        super().__init__()

        # Resize all inputs to match pretrained model standards (224×224)
        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

        # Stage 1: Initial convolution + stride-2 downsampling
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Stride-2 replaces MaxPool
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )

        # Residual block (32 → 64 channels)
        self.residual = SkipBlock(32, 64)

        # Stage 2: Deeper convolution + stride-2 downsampling
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Stride-2 downsampling
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )

        # Global Average Pooling layer (reduces H×W to 1×1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head: lightweight due to GAP
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flattens (B, C, 1, 1) → (B, C)
            nn.Linear(128, 64),  # Smaller FC layer since GAP reduces input size
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, number_of_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W)
        
        Returns:
            torch.Tensor: Class logits of shape (B, number_of_classes)
        """
        x = self.resize(x)     # Ensure input is 224×224
        x = self.stage1(x)     # Initial convolution + downsampling
        x = self.residual(x)   # Residual block
        x = self.stage2(x)     # Second convolution stage
        x = self.gap(x)        # Global average pooling
        return self.classifier(x)  # Classification head
