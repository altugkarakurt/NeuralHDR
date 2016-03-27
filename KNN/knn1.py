import numpy as np
import math
from scipy.spatial import distance
from MNIST_Database import MNIST

def knn(image, method='l3', k=3):
	train_samples = MNIST.data['train']
	np.random.shuffle(train_samples)
	image = MNIST.flatten_image(image)

	nearest_dist = [np.inf for _ in range(k)]
	nearest_lbls = [-1 for _ in range(k)]

	for sample in train_samples:
		candidate = MNIST.flatten_image(sample['image'])
		dist = distances(image, candidate, method)

		if dist < np.max(nearest_dist):
			max_idx = np.argmax(nearest_dist)
			nearest_dist[max_idx] = dist
			nearest_lbls[max_idx] = sample['label']

	return np.argmax(np.bincount(nearest_lbls))

def distances(a, b, method='euclidean'):
	if (method == 'manhattan'):
		return distance.minkowski(a, b, 1)
	elif (method == 'euclidean'):
		return distance.minkowski(a, b, 2)
	elif (method == 'l3'):
		return distance.minkowski(a, b, 3)
	elif (method == 'bhat'):
		return -math.log(sum(np.sqrt(a * b)))
	elif (method == 'intersection'):
		return len(a) / (sum(np.minimum(a, b)))
	elif (method == 'corr'):
		return 1.0 - np.correlate(a, b)
	else:
		return 0

test_confusion = np.zeros((10,10))
accuracy = 0

for sample in MNIST.data['test'][2250:4500]:
	y_est = knn(sample['image'])
	y = sample['label']
	test_confusion[y][y_est] += 1
	
	if y == y_est:
		accuracy += 1

np.set_printoptions(suppress=True)
print(confusion)
print('{} correct out of {}'.format(accuracy, 2250))


