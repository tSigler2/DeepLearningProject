import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

x = resnet50(weights=ResNet50_Weights.DEFAULT)

for param in x.parameters():
    x.requires_grad = False

x.fc = nn.Linear(2048, 12)
x.fc.requires_grad = True