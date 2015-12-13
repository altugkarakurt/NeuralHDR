import numpy as np
from numpy.random import randn
import math

from VCell import VCell

def crop(z, ri, ci, Al):
	x = (math.ceil(ri-Al[0]/2), math.ceil(ri+Al[0]/2))
	y = (math.ceil(ci-Al[1]/2), math.ceil(ci+Al[1]/2))
	res = np.zeros(Al)
	for i in range(*x):
		for j in range(*y):
			if (i >= 0) and (j >= 0) and (i < len(z)) and (j < len(z[0])):
				res[i+math.floor(Al[0]/2)-ri][j+math.floor(Al[1]/2)-ci] = z[i][j]
	return res

class VPlane:
	def __init__(self, K_cl, n_cl, n_sl, Al, c=None):
		self.n_sl = n_sl
		self.n_cl = n_cl
		self.K_cl = K_cl
		self.Al = Al

		self.c = c if c is not None else np.absolute(randn(*Al))

		self.cells = [[VCell(self.c) for _ in range(n_sl)] for _ in range(n_sl)]

	def feed_forward(self, u_cl):
		return np.array([[cell.feed_forward([crop(u_cl_i, ri, ci, self.Al) for u_cl_i in u_cl]) for ci, cell in enumerate(row)] for ri, row in enumerate(self.cells)])