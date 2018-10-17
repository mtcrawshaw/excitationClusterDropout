import sys
import os
import argparse
import importlib
import time

import torch
import torch.nn as nn
import torchvision.datasets as dset
from torch.autograd import Variable
import numpy as np

import utils
import architectures

parser = argparse.ArgumentParser(description='Train CNN models using Excitation Cluster Dropout on various datasets.')
parser.add_argument('dataset', help='Name of dataset to train/test on. List of options in data/datasets.txt')
parser.add_argument('network', help='Name of architecture to train. List of options in models/architectures.txt')
parser.add_argument('settingsFile', help='Path of settings file')
parser.add_argument('name', help='Name of experiment')
parser.add_argument('--reportFreq', type=int, default=50, help='Frequency of reporting loss and accuracy during training and validation.')
args = parser.parse_args()

rootDir = utils.rootDir()

# Create experiment directory
expName = args.name + "_" + time.strftime("%Y%m%d-%H%M%S")
expDir = os.path.join(rootDir, 'experiments', expName)
if os.path.isdir(expDir):
    print('Experiment directory already taken')
    sys.exit(1)
os.makedirs(expDir)
modelPath = os.path.join(expDir, args.name)
logPath = os.path.join(expDir, 'results.log')
os.mknod(logPath)

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
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = architectures.getNetwork(args.network, criterion)
    model = model.cuda()

    # Defining optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), settings.learningRate, momentum=settings.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(settings.epochs), eta_min=settings.learningRateMin)

    # Definiing training data
    trainTransform, validTransform = utils.dataTransforms(args.dataset, settings.cutout)
    trainData = dset.CIFAR10(root=settings.data, train=True, download=True, transform=trainTransform)
    numTrain = len(trainData)
    indices = list(range(numTrain))
    split = int(np.floor(settings.trainPortion * numTrain))

    trainQueue = torch.utils.data.DataLoader(
        trainData,
        batch_size=settings.batchSize,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True,
        num_workers=2)

    validQueue = torch.utils.data.DataLoader(
        trainData,
        batch_size=settings.batchSize,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
        pin_memory=True,
        num_workers=2)

    # Training
    for epoch in range(settings.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        utils.log(logPath, 'epoch ' + str(epoch) + ' lr ' + str(lr))

        trainingAccuracy = train(trainQueue, validQueue, model, criterion, optimizer, lr)
        # validAccuracy = infer(validQueue, model, criterion)
        utils.log(logPath, 'train acc ' + str(trainingAccuracy[0]) + ' ' + str(trainingAccuracy[1]))
        # utils.log(logPath, 'valid acc ' + str(validAccuracy[0]) + ' ' + str(validAccuracy[1]))

        torch.save(model.state_dict(), modelPath)

def train(trainQueue, validQueue, model, criterion, optimizer, lr):
    avgLoss = 0.0
    avgTop1 = 0.0
    avgTop5 = 0.0

    for step, (input, target) in enumerate(trainQueue):
        model.train()
        n = input.size(0)
		
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(async=True)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        if settings.gradClip is not None:
            nn.utils.clilp_grad_norm(model.parameters(), settings.gradClip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, (1, 5))
        avgLoss = (avgLoss * (step - 1) + loss.data[0] / n) / step
        avgTop1 = (avgTop1 * (step - 1) + prec1.data[0] / n) / step
        avgTop5 = (avgTop5 * (step - 1) + prec5.data[0] / n) / step

        if step % args.reportFreq == 0:
            utils.log(logPath, 'train ' + str(step) + ' ' + str(avgLoss) + ' ' + str(avgTop1) + ' ' + str(avgTop5))

    return avgTop1, avgTop5

def infer(validQueue, model, criterion):
    avgLoss = 0.0
    avgTop1 = 0.0
    avgTop5 = 0.0

    for step, (input, target) in enumerate(validQueue):
        n = input.size(0)

        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda()

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, (1, 5))
        avgLoss = (avgLoss * (step - 1) + loss.data[0] / n) / step
        avgTop1 = (avgTop1 * (step - 1) + prec1.data[0] / n) / step
        avgTop5 = (avgTop5 * (step - 1) + prec5.data[0] / n) / step

        if step % args.reportFreq == 0:
            utils.log(logPath, 'valid ' + str(step) + ' ' + str(avgLoss) + ' ' + str(avgTop1) + ' ' + str(avgTop5))

    return avgTop1, avgTop5

if __name__ == "__main__":
    main()

