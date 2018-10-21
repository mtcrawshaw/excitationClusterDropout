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
        self.conv3 = nn.Conv2d(48, 256, (4, 4))
        self.pool3 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, input, dropout=None):
        if dropout is not None:
            input_dropout = nn.Dropout(p=dropout['input'])
            conv_dropout = nn.Dropout(p=dropout['conv'])
            fc_dropout = nn.Dropout(p=dropout['fc'])

            input = input_dropout(input)

        # First convolutional layer
        x = self.conv1(input)
        x = F.relu(x)

        # First max pooling layer
        x = self.pool1(x)
        if dropout is not None:
            x = conv_dropout(x)

        # Second convolutional layer
        x = self.conv2(x)
        x = F.relu(x)

        # Second max pooling layer
        x = self.pool2(x)
        if dropout is not None:
            x = conv_dropout(x)

        # Third convolutional layer
        x = self.conv3(x)
        x = F.relu(x)

        # Third max pooling layer
        x = self.pool3(x)
        if dropout is not None:
            x = conv_dropout(x)

        # First fully connected layer
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = F.relu(x)
        if dropout is not None:
            x = fc_dropout(x)

        # Second fully connected layer
        x = self.fc2(x)
        x = F.relu(x)
        if dropout is not None:
            x = fc_dropout(x)

        # Third fully connected layer
        x = self.fc3(x)

        return x

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)
