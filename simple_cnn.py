"""
!pip install torch torchvision torchaudio --quiet
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

args = {}
args['img_size'] = 256
args['num_classes'] = 2
args['n_channel'] = 1

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        img_size = args['img_size']
        num_classes = args['num_classes']
        n_channel = args['n_channel']
        # Define the layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        
        # Output size after conv2 and pool2: (64 channels, 64 height, 64 width)
        input_size = 64 * 64 * 64 

        self.fc1 = nn.Linear(input_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
