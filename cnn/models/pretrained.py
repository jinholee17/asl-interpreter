import torch.nn as nn
from torchvision import models


def get_pretrained_resnet(num_classes):
    """
    uses ResNet which is a already pre trained model but will insteas fine tune the last layer for our purposes    
    Returns:
        model (torch.nn.Module): the fien tuned ResNet model
    """
    model = models.resnet18(pretrained=True)  # load pre resnet18

    # keep previous layers 
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features  
    model.fc = nn.Linear(num_features, num_classes)  # replace final layer

    return model



