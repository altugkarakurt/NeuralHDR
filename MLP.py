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
		x_l = data

		# Iterates over layers
		for l, l_size in enumerate(self.sizes[1:]):
			y_l = np.zeros(l_size)

			# Iterates over perceptrons of current layer
			for i in range(l_size):
				cur_weights = self.weights[l][i] + self.biases[l][i]
				bias_x = [1]
				x_l = bias_x.extend(x_l)
				local_field = np.dot(self.weights[l][i], x_l) + self.biases[l]
				y_l[i] = self.activation_function(local_field)

			# Input of l^th layer is the output of (l-1)^st layer
			x_l = y_l
		return y_l

	def train(self, labels, samples, epochs, m, eta):
		training_data = list(zip(samples, np.array(labels)))
		n = len(training_data)

		for epoch_idx in range(epochs):
			shuffle(training_data)
			mini_batches = [training_data[k:k+m] for k in range(0, n, m)]

			for idx, mini_batch in enumerate(mini_batches):
				self.train_mini_batch(mini_batch, eta)
				print('Mini-batch %d/%d of Epoch %d completed.\n' %(batch_idx, len(mini_bacthes), epoch_idx))


	def train_mini_batch(self, mini_batch, eta):
		grad_b = [np.zeros(b.shape) for b in self.biases]
		grad_w = [np.zeros(w.shape) for w in self.weights]

		for data, label in mini_batch:
			delta_grad_b, delta_grad_w = self.back_propagation(data, label)
			grad_b = [nb+dnb for nb, dnb in list(zip(grad_b, delta_grad_b)])
			grad_w = [nw+dnw for nw, dnw in list(zip(grad_w, delta_grad_w)])

		self.weights = [w-(eta/len(mini_batch)) * nw for w, nw in list(zip(self.weights, grad_w)])
		self.biases = [b-(eta/len(mini_batch)) * nb for b, nb in list(zip(self.biases, grad_b)])

	def back_propagation(self, data, label):
		grad_b = [np.zeros(b.shape) for b in self.biases]
		grad_w = [np.zeros(w.shape) for w in self.weights]
		
		# feedforward
		activation = data
		activations = [data] # list to store all the activations, layer by layer
		zs = [] # list to store all the z vectors, layer by layer
		for b, w in list(zip(self.biases, self.weights)):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		
		delta = (activations[-1] - label) * d_sigmoid(zs[-1])
		grad_b[-1] = delta
		grad_w[-1] = np.dot(delta, activations[-2].transpose())

		for l in range(2, len(self.sizes)):
			z = zs[-l]
			sp = d_sigmoid(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			grad_b[-l] = delta
			grad_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (grad_b, grad_w)