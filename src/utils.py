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

	valid_transform = transforms.Compose([
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
