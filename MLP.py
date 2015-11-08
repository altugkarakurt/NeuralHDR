import numpy as np
from numpy.random import randn
import random

class MLP:

	def __init__(self, layer_sizes):
		self.layer_sizes = layer_sizes
		self.biases = [randn(y, 1) for y in sizes[1:]]
		self.weights = [randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

	def estimate(self, digit):

	def train(self, labels, digits):