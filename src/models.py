from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def create_resnet18(num_classes: int = 2) -> nn.Module:
    """Create an ImageNet-pretrained ResNet-18 with partial fine-tuning."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in model.conv1.parameters():
        param.requires_grad = False

    for param in model.bn1.parameters():
        param.requires_grad = False

    for param in model.layer1.parameters():
        param.requires_grad = False

    for param in model.layer2.parameters():
        param.requires_grad = False

    for param in model.layer3.parameters():
        param.requires_grad = False
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
