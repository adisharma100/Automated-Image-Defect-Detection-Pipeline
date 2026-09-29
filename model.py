import torch.nn as nn
import torchvision
from .config import CLASSES


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class DefectCNN(nn.Module):
    """Compact 4-block CNN for 1-channel inputs."""
    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32), conv_block(32, 64), conv_block(64, 128), conv_block(128, 256))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def resnet18_transfer(num_classes=len(CLASSES), pretrained=True):
    """Transfer-learning alternative (1-channel input adapted from RGB weights)."""
    w = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    m = torchvision.models.resnet18(weights=w)
    old = m.conv1
    m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    if pretrained:
        m.conv1.weight.data = old.weight.data.sum(dim=1, keepdim=True)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def get_model(name="cnn"):
    return resnet18_transfer() if name == "resnet18" else DefectCNN()
