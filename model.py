import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import utils
import pdb
from torchsummary import summary
import torch
import torch.utils.model_zoo as model_zoo
import numpy as np
class AmazonSimpleNet(nn.Module):
    """Simple convnet """
    def __init__(self):
        super(AmazonSimpleNet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 17),
        )

    def forward(self, x):
        x = x.float()
        x = self.features(x)
        x = x.view(x.size(0), 256 * 8 * 8)
        x = self.classifier(x)
        return F.sigmoid(x)

class AmazonNIRResNet18(nn.Module):

    """ Resnet 18 pretrained"""

    def __init__(self):
        super(AmazonResNet18,self).__init__()
        # Mean of weights:
        model = models.resnet18(num_classes=17)

        RESNET_18 = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
        state = model_zoo.load_url(RESNET_18)
        mean_weights = torch.mean(state['conv1.weight'],dim=3, keepdim=True)


        self.pretrained_model = models.resnet18(pretrained=True)
        shape_weights = np.shape(self.pretrained_model.conv1.weight)
        random_weights = torch.rand(shape_weights[0],shape_weights[1],shape_weights[2],shape_weights[3])

        #Para cuatro canales:
        self.pretrained_model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=1,padding=(3, 3), bias=False)
        
        with torch.no_grad():
            self.pretrained_model.conv1.weight[:, :3] = mean_weights
            self.pretrained_model.conv1.weight[:, 3] = self.pretrained_model.conv1.weight[:, 0]
    
        #x = torch.randn(10, 4, 224, 224)
        #output = model(x)

        #new_input_conv.weight = nn.Parameter(new_input_conv.weight.detach().requires_grad_(True))

        classifier = [
            nn.Linear(self.pretrained_model.fc.in_features, 17)
        ]
        self.classifier = nn.Sequential(*classifier)
        self.pretrained_model.fc = self.classifier
    
    def forward(self, x):
        x = x.float()
        x = self.pretrained_model(x)
        return F.sigmoid(x)

class AmazonResNet18(nn.Module):

    """ Resnet 18 pretrained"""

    def __init__(self):
        super(AmazonResNet18,self).__init__()
        self.pretrained_model = models.resnet18(pretrained=True)
        classifier = [
            nn.Linear(self.pretrained_model.fc.in_features, 17)
        ]
        self.classifier = nn.Sequential(*classifier)
        self.pretrained_model.fc = self.classifier

    def forward(self, x):
        x = x.float()
        x = self.pretrained_model(x)
        return F.sigmoid(x)
        
class AmazonNIRResNet18(nn.Module):

    """ Resnet 18 pretrained"""

    def __init__(self):
        super(AmazonNIRResNet18,self).__init__()
        self.pretrained_model = models.resnet18(pretrained=True)
        classifier = [
            nn.Linear(self.pretrained_model.fc.in_features, 17)
        ]
        self.classifier = nn.Sequential(*classifier)
        self.pretrained_model.fc = self.classifier
    
    def forward(self, x):
        x = x.float()
        x = self.pretrained_model(x)
        return F.sigmoid(x)

class AmazonResNet101(nn.Module):

    """ Resnet 101 pretrained"""

    def __init__(self):
        super(AmazonResNet101,self).__init__()
        # Mean of weights:
        model = models.resnet101(num_classes=17)

        RESNET_101 = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
        state = model_zoo.load_url(RESNET_101)
        mean_weights = torch.mean(state['conv1.weight'],dim=3, keepdim=True)


        self.pretrained_model = models.resnet101(pretrained=True)
        #Para cuatro canales:
        self.pretrained_model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=1,padding=(3, 3), bias=False)
        
        with torch.no_grad():
            self.pretrained_model.conv1.weight[:, :3] = mean_weights
            self.pretrained_model.conv1.weight[:, 3] = self.pretrained_model.conv1.weight[:, 0]
    
        #x = torch.randn(10, 4, 224, 224)
        #output = model(x)

        #new_input_conv.weight = nn.Parameter(new_input_conv.weight.detach().requires_grad_(True))

        classifier = [
            nn.Linear(self.pretrained_model.fc.in_features, 17)
        ]
        self.classifier = nn.Sequential(*classifier)
        self.pretrained_model.fc = self.classifier

        
    def forward(self, x):
        x = x.float()
        x = self.pretrained_model(x)
        return F.sigmoid(x)

class AmazonInceptionV3(nn.Module):

    """ Inception V3 pretrained"""

    def __init__(self):
        super(AmazonInceptionV3,self).__init__()
        INCEPTION_V3 = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        self.pretrained_model = models.inception_v3(num_classes=17,aux_logits=False)
        state = model_zoo.load_url(INCEPTION_V3)
        state = {x: state[x] for x in state if not x.startswith('fc')}
        state = {x: state[x] for x in state if not x.startswith('AuxLogits')}
        model_state = self.pretrained_model.state_dict()
        model_state.update(state)
        self.pretrained_model.load_state_dict(model_state)
        """
        classifier = [
            nn.Linear(self.pretrained_model.fc.in_features, 17)
        ]
        self.classifier = nn.Sequential(*classifier)
        self.pretrained_model.fc = self.classifier
        """
    def forward(self, x):
        x = x.float()
        x = self.pretrained_model(x)
        return F.sigmoid(x)

if __name__ == '__main__':
    net = AmazonSimpleNet().cuda()
    size = utils.calculate_feature_size(net.features,(224,224))
    print(size)
    summary(net, (3, 256, 256))


