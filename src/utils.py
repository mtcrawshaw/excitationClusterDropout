import os

import torchvision.transforms as transforms

class Cutout(object):
	def __init__(self, length):
		self.length = length

	def __call__(self, img):
		h, w = img.size(1), img.size(2)
		mask = np.ones((h, w), np.float32)
		y = np.random.randint(h)
		x = np.random.randint(w)

		y1 = np.clip(y - self.length // 2, 0, h)
		y2 = np.clip(y + self.length // 2, 0, h)
		x1 = np.clip(x - self.length // 2, 0, w)
		x2 = np.clip(x + self.length // 2, 0, w)

		mask[y1:y2, x1:x2] = 0.
		mask = torch.from_numpy(mask)
		mask = mask.expand_as(img)
		img *= mask
		return img


def dataTransforms(dataset, cutout):
	mean, std = distribution(dataset)

	trainTransform = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)])
	
	if cutout is not None:
		trainTransform.transforms.append(Cutout(cutout))

	validTransform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean, std)])

	return trainTransform, validTransform

def distribution(dataset):
	mean = None
	std = None

	if dataset == 'cifar10':
		mean = [0.49139968, 0.48215827, 0.44653124]
		std = [0.24703233, 0.24348505, 0.26158768]

	return mean, std

def accuracy(logits, target, topk):
	maxk = max(topk)
	batchSize = target.size(0)

	_, pred = output.logits(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correctK = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batchSize))
	return res

def rootDir():
	return os.path.dirname(os.path.dirname(__file__))

def save(stateDict, modelPath):
	torch.save(stateDict, modelPath)

def log(logPath, msg):
	with open(logPath) as f:
		f.write(msg + '\n')
