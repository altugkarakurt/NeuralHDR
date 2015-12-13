import numpy as np
from numpy.random import randn
import math

from SLayer import SLayer
from CLayer import CLayer

def windower(pl_size, l_size):
	m = abs(pl_size-l_size)+1
	return (m, m)

class Neocognitron:
	def __init__(self, input_size, layer_sizes, windows=None, a=None, b=None, c=None, d=None, r=None, sigma=None):
		self.input_size = input_size
		self.layer_sizes = layer_sizes
		self.windows = windows if windows is not None else \
			[windower(layer_sizes[l_ind][0], layer_sizes[l_ind-1][0]) if l_ind > 0 \
			else windower(layer_sizes[l_ind][0], input_size) \
			for l_ind, _ in enumerate(layer_sizes)]

		self.a = a if a is not None else np.array([None for _ in range(int(len(layer_sizes)/2))])
		self.b = b if b is not None else np.array([None for _ in range(int(len(layer_sizes)/2))])
		self.c = c if c is not None else np.array([None for _ in range(int(len(layer_sizes)/2))])
		self.d = d if d is not None else np.array([None for _ in range(int(len(layer_sizes)/2))])
		self.r = r if r is not None else np.array([None for _ in range(int(len(layer_sizes)/2))])
		self.sigma = sigma if sigma is not None else np.array([None for _ in range(int(len(layer_sizes)/2))])

		self.layers = []

		for l_ind, l_size in enumerate(layer_sizes):
			# Odd indices, C-Layers
			if l_ind % 2:
				K_cl = l_size[1]
				# K_sl = layer_sizes[l_ind-1][1]
				n_cl = l_size[0]
				n_sl = layer_sizes[l_ind-1][0]
				Dl = self.windows[l_ind]
				di = self.d[math.floor(l_ind/2.0)]
				self.layers.append(CLayer(K_cl, n_cl, n_sl, Dl, di))
			
			# Even indices, S-Layers
			else:
				K_cl = layer_sizes[l_ind-1][1] if l_ind > 0 else 1
				K_sl = l_size[1]
				n_cl = layer_sizes[l_ind-1][0] if l_ind > 0 else input_size
				n_sl = l_size[0]
				Al = self.windows[l_ind]
				ai = self.a[int(l_ind/2)]
				bi = self.b[int(l_ind/2)]
				ci = self.c[int(l_ind/2)]
				ri = self.r[int(l_ind/2)]
				sigmai = self.sigma[int(l_ind/2)]
				self.layers.append(SLayer(K_cl, K_sl, n_cl, n_sl, Al, ai, bi, ci, ri, sigmai))

	def feed_forward(self, image):
		u_last = [np.array(image)]
		for l_ind, layer in enumerate(self.layers):
			# print("Layer {} begin.".format(l_ind))
			u_last = layer.feed_forward(u_last)
		return u_last
		#while(l < len(self.layers) - 1):
		#	return feedforward(l-1)