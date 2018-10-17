import torch
import torch.nn as nn
import torch.nn.functional as F

def getNetwork(networkName, criterion):
    if networkName == 'cnn2':
        return cnn2Network(criterion)

class cnn2Network(nn.Module):
    def __init__(self, criterion):
        super(cnn2Network, self).__init__()
        self._criterion = criterion
        self.conv1 = nn.Conv2d(3, 32, (5, 5))
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 48, (5, 5))
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(48, 256, (5, 5))
        self.pool3 = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(7 * 7 * 48, 2048)
        self.fc2 = nn.Linear(2048, 1048)
        self.fc3 = nn.Linear(1048, 10)

    def forward(self, input):
        print(input.shape)
        # First convolutional layer
        x = self.conv1(input)
        x = F.relu(x)
        print(x.shape)

        # First max pooling layer
        x = self.pool1(x)
        print(x.shape)

        # Second convolutional layer
        x = self.conv2(x)
        x = F.relu(x)
        print(x.shape)

        # Second max pooling layer
        x = self.pool2(x)
        print(x.shape)

        # Third convolutional layer
        x = self.conv3(x)
        x = F.relu(x)
        print(x.shape)

        # Third max pooling layer
        x = self.pool3(x)
        print(x.shape)

        # First fully connected layer
        x = self.fc1(x)
        x = F.relu(x)
        print(x.shape)

        # Second fully connected layer
        x = self.fc2(x)
        x = F.relu(x)
        print(x.shape)

        # Third fully connected layer
        x = self.fc3(x)
        print(x.shape)

        return x

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)
