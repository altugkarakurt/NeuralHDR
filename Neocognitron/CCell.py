import numpy as np
from random import randint
from numpy.random import randn

def phi(x): return x if x > 0 else 0

class CCell:
	def __init__(self, weights, alpha=None):
		self.alpha = alpha if alpha is not None else randn()
		self.weights = weights

	def feed_forward(self, v_in):
		temp = (1 + self.weights @ v_in)/(1 + v_in) - 1
		return phi(temp/(self.alpha + temp))