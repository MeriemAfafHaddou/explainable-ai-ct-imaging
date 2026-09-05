import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def create_resnet18(num_classes: int = 2) -> nn.Module:
    """Create an ImageNet-pretrained ResNet-18 for classification."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model
