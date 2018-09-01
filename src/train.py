import sys
import os
import argparse
import importlib

import torch
import torch.nn as nn
import torchvision.datasets as dset

import utils
import architectures

parser = argparse.ArgumentParser(description='Train CNN models using Excitation Cluster Dropout on various datasets.')
parser.add_argument('settingsFile', help='Path of settings file')
parser.add_argument('dataset', help='Name of dataset to train/test on. List of options in data/datasets.txt')
parser.add_argument('network', help='Name of architecture to train. List of options in models/architectures.txt')
args = parser.parse_args()

def main():
	# Check that GPU is available
	if not torch.cuda.is_available():
		print('No gpu device available')
		sys.exit(1)

	# Parse settings file
	sys.path.append(os.path.dirname(args.settingsFile))
	settingsName = os.path.basename(args.settingsFile[:-3])
	settings = importlib.import_module(os.path.basename(settingsName))

	# Defining loss and model
	loss = nn.CrossEntropyLoss()
	loss = loss.cuda()
	model = architectures.getNetwork(args.network, loss)
	model = model.cuda()

	# Defining optimizer and scheduler
	optimizer = torch.optim.SGD(model.parameters(), settings.learningRate, momentum=settings.momentum)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(settings.epochs), eta_min=settings.learningRateMin)

	# Definiing training data
	trainTransform, validTransform = utils.dataTransforms(args.dataset, settings.cutout)
	trainData = dset.CIFAR10(root=args.data, train=True, download=True, transform=trainTransform)
	numTrain = len(trainData)
	indices = list(range(numTrain))
	split = int(np.floor(args.trainPortion * numTrain))

	trainQueue = torch.utils.data.DataLoader(
		trainData,
		batch_size=args.batch_size,
		sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
		pin_memory=True,
		num_workers=2)

	validQueue = torch.utils.data.DataLoader(
		trainData,
		batch_size=args.batch_size,
		sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
		pin_memory=True,
		num_workers=2)

if __name__ == "__main__":
	main()

