import numpy as np
from numpy.random import randn
import random
from numpy import sign as signum

def heaviside(x): return (1 if x >= 0 else 0)

class MLP:

	def __init__(self, layer_sizes, activation_function=heaviside, biases=None, weights=None):
		# TODO: check whether bias, weight sizes are compatible with layer sizes
		self.layer_sizes = layer_sizes
		self.biases = biases if biases is not None else [randn(y, 1) for y in sizes[1:]] 
		self.weights = weights if weights is not None else [randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
		self.activation_function = activation_function


	def __call__(self, inputs):
		x_l = inputs

		# Iterates over layers
		for l, l_size in enumerate(self.layer_sizes[1:]):
			y_l = np.zeros(l_size)

			# Iterates over perceptrons of current layer
			for i in xrange(l_size):
				local_field = np.dot(self.weights[l][i], x_l) + self.biases[l]
				y_l[i] = self.activation_function(local_field)

			# Input of l^th layer is the output of (l-1)^st layer
			x_l = y_l
		return y_l

	
