import torch.nn as nn
import torch.nn.functional as F


class ASLCNN(nn.Module):
    def __init__(self, num_classes):
        super(ASLCNN, self).__init__()
        # conv
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # output: 32x128x128
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # output: 64x128x128
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # half size: 64x64x64

        # fully connected
        self.flatten_size = 64 * 64 * 64  #  channels * height * width after pooling
        self.fc1 = nn.Linear(self.flatten_size, 128)  # flatten to 128 neurons
        self.fc2 = nn.Linear(128, num_classes)  # output to `num_classes`


    def forward(self, x):
        x = F.relu(self.conv1(x))  # conv +relu
        x = self.pool(F.relu(self.conv2(x)))  # conv relu pool
        x = x.view(x.size(0), -1)  # flatten tensor
        x = F.relu(self.fc1(x))  # fully connected relu
        x = self.fc2(x)  # output
        return x




