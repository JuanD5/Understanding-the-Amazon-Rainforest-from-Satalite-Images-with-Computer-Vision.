import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import utils


class PlanetResNet18(nn.Module):

    """ Resnet 18 pretrained"""

    def __init__(self):
        super().__init__()
        self.pretrained_model = models.resnet18(pretrained=True)
        classifier = [
            nn.Linear(self.pretrained_model.fc.in_features, 17)
        ]
        self.classifier = nn.Sequential(*classifier)
        self.pretrained_model.fc = self.classifier

    def forward(self, x):
        x = self.pretrained_model(x)
        return F.sigmoid(x)
        