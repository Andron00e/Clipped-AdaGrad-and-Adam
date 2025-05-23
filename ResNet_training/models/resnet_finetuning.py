import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention

class ResNet18ToFinetune(nn.Module):
    """
    A PyTorch neural network module that adapts the standard ResNet-18 architecture for fine-tuning on CIFAR-10 images.
    This class modifies the initial convolutional layer of ResNet-18 to accept 32x32 images (as in CIFAR-10) by changing the kernel size, stride, and padding. 
    It also replaces the final fully connected layer to output logits for 10 classes.
    Attributes:
        resnet18 (torchvision.models.ResNet): The modified ResNet-18 model.
    Methods:
        forward(x):
            Performs a forward pass through the network and returns an ImageClassifierOutputWithNoAttention object containing the logits.
    Example:
        model = ResNet18ToFinetune()
        output = model(images)
    """

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

