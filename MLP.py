import numpy as np
from numpy.random import randn
import random

class MLP:

	def __init__(self, layer_sizes):
		self.layer_sizes = layer_sizes
		self.biases = [randn(y, 1) for y in sizes[1:]]
		self.weights = [randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

	def estimate(self, digit):
		x_l = digit
		for l, l_size in enumerate(layer_sizes[1:]):
			y_l = np.zeros(l_size)
			for i in xrange(l_size)
				local_field = np.dot(self.weights[l][i], x_l) + self.biases[l]
				x_l[i] = activation_function(local_field)

	def train(self, labels, digits):