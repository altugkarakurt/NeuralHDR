import numpy as np
import random
import pdb

from numpy.random import randn
from random import shuffle

def heaviside(x): return (1 if x >= 0 else 0)
def d_heaviside(x): return (1 if x == 0 else 0)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def d_sigmoid(x): return sigmoid(x) * (1 - sigmoid(x))


class MLP:

	def __init__(self, sizes, activation_function=(sigmoid, d_sigmoid), biases=None, weights=None):
		# TODO: check whether bias, weight sizes are compatible with layer sizes
		self.sizes = sizes
		self.biases = np.array(biases) if biases is not None else [randn(y, 1) for y in sizes[1:]] 
		self.weights = np.array(weights) if weights is not None else [randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
		self.activation_function = activation_function[0]
		self.d_activation_function = activation_function[1]


	def __call__(self, data):
		self.estimate(data)

	def estimate(self, data):
		raise NotImplementedError

	def train(self, labels, samples, epochs, m, eta):
		raise NotImplementedError

	def train_mini_batch(self, mini_batch, eta):
		raise NotImplementedError

	def back_propagation(self, data, label):
		raise NotImplementedError
	