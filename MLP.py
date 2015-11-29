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

	def __init__(self, sizes, activation_function=(sigmoid, d_sigmoid), weights=None):
		# TODO: check whether bias, weight sizes are compatible with layer sizes
		# Remember weights[layer][0] are the biases
		self.sizes = np.array(sizes)
		self.weights = np.array(weights) if weights is not None else [randn(y, x) for x, y in \
			zip(np.concatenate(([sizes[0]], sizes[:-1])) + np.ones_like(sizes), sizes)]
		self.activation_function = activation_function[0]
		self.d_activation_function = activation_function[1]


	def __call__(self, network_input):
		return self.estimate(network_input)

	def estimate(self, network_input):
		# consider the input as output of the -1st layer
		layer_output = np.array(network_input)
		
		for layer, layer_size in enumerate(self.sizes):
			# [1] is the "bias neuron"
			layer_input = np.concatenate(([1], layer_output))
			layer_output = np.zeros(layer_size)
			
			for neuron in range(layer_size):
				local_field = layer_input @ self.weights[layer][neuron]
				layer_output[neuron] = self.activation_function(local_field)
		
		return layer_output

	def train(self, labels, samples, epochs, m, eta):
		raise NotImplementedError

	def train_mini_batch(self, mini_batch, eta):
		raise NotImplementedError

	def back_propagation(self, data, label):
		raise NotImplementedError
	