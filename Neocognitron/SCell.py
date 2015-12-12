import numpy as np
from random import randint
from numpy.random import randn

def phi(x): return x if x > 0 else 0

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