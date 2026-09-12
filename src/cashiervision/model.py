

from torch import nn
from torchvision.models import ResNet, ResNet18_Weights, resnet18


def build_model(num_classes: int, pretrained: bool = True) -> ResNet:
    """ Using ResNet model but just altering a layer

    """
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    # Replacing final layer since we only have 18 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
