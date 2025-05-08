import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention

class ResNet18ToFinetune(nn.Module):
    def __init__(self):
        super().__init__()

        weights = torchvision.models.ResNet18_Weights
        self.resnet18 = torchvision.models.resnet18(weights=weights)
        
        # to make it possible to handle cifar10 32x32 images
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.resnet18.fc = nn.Linear(512, 10, bias=True)
        
    def forward(self, x):
        out = self.resnet18(x)
        return ImageClassifierOutputWithNoAttention(
            logits=out,
        )

