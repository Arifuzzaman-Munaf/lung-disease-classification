import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T



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
    Residual block with two convolutional layers, batch norm, and skip connection.
    Keeps spatial dimensions unchanged.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # First conv layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        # Second conv layer: same output channels
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        return x1 + x2  # Residual connection


class CustomResidualCNN(nn.Module):
    """
    Custom CNN model with skip connections for classification.
    Includes resizing to 224×224 to match pretrained standards.
    """
    def __init__(self, number_of_classes= 8):
        super().__init__()
        # Resize input to fixed size

        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

        # Stage 1: initial conv, ReLU, batch norm and downsampling
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2) 
        )

        # Residual block: increases channels to 64
        self.residual = SkipBlock(32, 64)
        self.pool1 = nn.MaxPool2d(2)  

        # Stage 2: deeper conv stage with 128 channels
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)  
        )

        # Infer feature size dynamically with a dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            x = self.stage1(dummy)
            x = self.residual(x)
            x = self.pool1(x)
            x = self.stage2(x)
            n_features = x.numel() // x.shape[0]

        # Fully connected classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, number_of_classes)
        )

    def forward(self, x):
        x = self.resize(x)       # Ensure consistent size
        x = self.stage1(x)       # First conv stage
        x = self.residual(x)     # Residual block
        x = self.pool1(x)        # Pooling
        x = self.stage2(x)       # Second conv stage
        return self.classifier(x)  # Fully connected head
