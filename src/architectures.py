import torch
import torch.nn as nn

def getNetwork(networkName, criterion):
	if networkName == 'cnn2':
		return cnn2Network(criterion)

class cnn2Network(nn.Module):
	def __init__(self, criterion):
		super(cnn2Network, self).__init__()
		self._criterion = criterion
		self.classifier = nn.Linear(3, 10)

	def forward(self, input):
		return classifier(input)

	def _loss(self, input, target):
		logits = self(input)
		return self._criterion(logits, target)		
