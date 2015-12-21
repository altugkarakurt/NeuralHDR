import numpy as np
from numpy.random import randn, rand
import math

from SLayer import SLayer
from CLayer import CLayer
from VCPlane import VCPlane

class Neocognitron:
	#def __init__(self, input_size, layer_sizes, windows=None, a=None, b=None, c=None, d=None, r=None, sigma=None):
	def __init__(self, input_size, layer_sizes, windows, a=None, b=None, c=None, d=None, r=None):
		self.input_size = input_size
		self.layer_sizes = layer_sizes
		self.windows = windows

		cs_count = int(len(layer_sizes)/2)

		gamma = [0.11, 0.42, 0.21, 0.06]
		delta = [0.5, 0.8, 0.4, 0.5]
		deltahat = [0.1, 0.06, 0.42, 0.78]


		self.a = a if a is not None else np.array([ \
			self.gen_a(1, layer_sizes[0][1], windows[0])
		] + [ \
			self.gen_a(layer_sizes[2*cs_ind-1][1], layer_sizes[2*cs_ind][1], windows[2*cs_ind]) \
			for cs_ind in range(1, cs_count) \
		])

		# self.a = a if a is not None else np.array([ \
		# 	[[rand(*windows[0]) \
		# 	for _ in range(1)] \
		# 	for _ in range(layer_sizes[0][1])]
		# ] + [ \
		# 	[[rand(*windows[2*cs_ind]) \
		# 	for _ in range(layer_sizes[2*cs_ind-1][1])] \
		# 	for _ in range(layer_sizes[2*cs_ind][1])]
		# 	for cs_ind in range(1, cs_count) \
		# ])

		self.b = b if b is not None else np.array([ \
			[0 for _ in range(layer_sizes[2*cs_idx][1])] \
			for cs_idx in range(cs_count) \
		])

		self.c = c if c is not None else np.array([ \
			self.gen_c(*windows[2*cs_idx+2], gamma[cs_idx]) \
			for cs_idx in range(cs_count) \
		])

		self.r = r if r is not None else np.array([4.5, 1.5, 1.5, 2.5])

		# self.r = r if r is not None else np.array([ \
		# 	np.absolute(randn()) for _ in range(cs_count) \
		# ])

		self.d = d if d is not None else np.array([ \
			self.gen_d(*windows[2*cs_idx+1], delta[cs_idx], deltahat[cs_idx]) \
			for cs_idx in range(cs_count) \
		])

		self.layers = []

		for l_ind, l_size in enumerate(layer_sizes):
			is_c_layer = l_ind % 2
			cs_idx = int((l_ind - is_c_layer) / 2)

			# Odd indices, C-Layers
			if is_c_layer:
				n_cl = l_size[0]
				K_cl = l_size[1]
				Dli = windows[l_ind]
				Sli = windows[l_ind + 1]
				ci = self.c[cs_idx]
				di = self.d[cs_idx]
				self.layers.append(CLayer(K_cl, n_cl, Dli, Sli, ci, di))
			
			# Even indices, S-Layers
			else:
				n_sl = l_size[0]
				K_sl = l_size[1]
				n_cl = layer_sizes[l_ind-1][0] if l_ind != 0 else input_size
				K_cl = layer_sizes[l_ind-1][1] if l_ind != 0 else 1
				Sli = windows[l_ind]
				Dli = windows[l_ind + 1]
				ai = self.a[cs_idx]
				bi = self.b[cs_idx]
				di = self.d[cs_idx]
				ri = self.r[cs_idx]
				self.layers.append(SLayer(K_cl, K_sl, n_cl, n_sl, Sli, Dli, ai, bi, di, ri))

		self.vplane = VCPlane(input_size, windows[0])

	def estimate(self, image):
		return self.feed_forward(image)[-1][0]

	def gen_a(self, K_cl, K_sl, Sl):
		return [[np.absolute(randn(*Sl)) for _ in range(K_cl)] for _ in range(K_sl)]

	def gen_c(self, a, b, gamma):
		dist = lambda x,y : math.sqrt((x - a/2) ** 2 + (y - b/2) ** 2)
		return [[gamma ** dist(i, j) for i in range(a)] for j in range(b)]

	def gen_d(self, a, b, delta, deltahat):
		dist = lambda x,y : math.sqrt((x - a/2) ** 2 + (y - b/2) ** 2)
		return [[(delta ** dist(i, j)) * deltahat for i in range(a)] for j in range(b)]

	def feed_forward(self, image):
		u_last = [np.array(image)]
		v_last = self.vplane.feed_forward(u_last)
		uv_list = [(u_last, v_last)]
		for l_ind, layer in enumerate(self.layers):
			print("Layer {} begin.".format(l_ind))
			(u_last, v_last) = layer.feed_forward(u_last, v_last)
			uv_list.append((u_last, v_last))
		return uv_list

	def train(self, data, q_l):
		pass

