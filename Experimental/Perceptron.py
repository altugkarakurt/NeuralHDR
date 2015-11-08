# -*- coding: utf-8 -*-

import numpy
from numpy import sign as signum

# def heaviside(x): return (0.5 * (numpy.sign(x) + 1))
def heaviside(x): return (1 if x >= 0 else 0)

class Perceptron:
	def __init__(self, *weights, activation_function=heaviside):
		self.activation_function = activation_function
		self.weights = numpy.array(weights)
	
	def __call__(self, *inputs):
		inputs = numpy.array(inputs)
		local_field = self.weights @ inputs;
		return self.activation_function(local_field)