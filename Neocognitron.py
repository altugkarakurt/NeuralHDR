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

class SCell:
	def __init__(self, weights, r=None):
		self.r = r if r is not None else randn()
		self.weights = weights

	def feed_forward(self, window_in, v_in, inhibit):
		# window_in: input window vector
		# v_n:		  input fed from preceding v-cell
		# inhibit:   inhibitory weight from V cell  

		temp = (1 + (self.weights @ window_in)) / (1 + 2 * self.r / (1+self.r) * inhibit * v_in) - 1
		return r * phi(temp)
	}

class Neocognitron:

	def __init__(self, s_sizes=None, c_sizes=None, weights=None):
		# all planes are assumed to be square
		self.s_sizes = s_sizes if s_sizes is not None else [16, 8, 2]
		self.c_sizes = s_sizes if c_sizes is not None else [10, 6, 1]
		
		self.gamma = [randint(0,1) for _ in self.c_sizes]
		self.delta = [randint(0,1)*0.5 + 0.5 for _ in self.c_sizes] # interval: [0.5, 1]
		self.delta_bar = [randint(0,1) for _ in self.c_sizes]
