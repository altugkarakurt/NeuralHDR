import numpy as np
import random

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
		return self.feed_forward(network_input)[-1]
	
	def feed_forward(self, network_input):
		# consider the input as output of the 0th layer
		layer_outputs = np.array([np.zeros(size) for size in self.sizes])
		
		for layer, layer_size in enumerate(self.sizes):
			# [1] is the "bias neuron"
			layer_input = np.concatenate(([1], layer_outputs[layer-1])) if layer != 0 \
				else np.concatenate(([1], np.array(network_input)))
			
			for neuron in range(layer_size):
				local_field = layer_input @ self.weights[layer][neuron]
				layer_outputs[layer][neuron] = self.activation_function(local_field)
		
		return layer_outputs


	def train(self, data, labels, epochs, block_size, learn_rate):
		training_data = list(zip(data, np.array(labels)))
		training_size = len(training_data)
		
		for epoch in range(epochs):
			shuffle(training_data)
			blocks = [training_data[k:k+block_size] for k in range(0, training_size, block_size)]
			
			for block_idx, block in enumerate(blocks):
				self.train_block(block, learn_rate)
				print("Block %d/%d of Epoch %d complete.\n" % (block_idx + 1, len(blocks), epoch + 1))


	def train_block(self, block, learn_rate):
		gradient = np.array([np.zeros_like(w) for w in self.weights])
		
		for data, label in block:
			gradient += self.back_propagation(data, label)
		
		# gradient /= len(block)
		self.weights -= learn_rate * gradient


	def back_propagation(self, data, label):
		raise NotImplementedError
	