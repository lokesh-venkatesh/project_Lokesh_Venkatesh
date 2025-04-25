import torch.nn as nn
from torchvision import models

class ChestXRayModel(nn.Module):
    """
    A DenseNet-121-based CNN for chest X-ray classification.
    
    - Accepts grayscale (1-channel) input of size 224x224.
    - No pretrained weights (trained from scratch).
    - Outputs predictions for 4 classes.

    Args:
        num_classes (int): Number of output classes (default: 4).
    """
    def __init__(self, num_classes=4):
        super(ChestXRayModel, self).__init__()

        # Load DenseNet-121 without pretrained weights
        self.base_model = models.densenet121(pretrained=False)

        # Modify the first convolution layer to accept grayscale input
        self.base_model.features.conv0 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Replace the classifier for num_classes
        self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, num_classes)

        # Initialize weights from scratch
        self._initialize_weights()

    def forward(self, x):
        return self.base_model(x)

    def _initialize_weights(self):
        """
        Re-initialize all layers with standard initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)