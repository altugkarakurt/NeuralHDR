import numpy as np
from numpy.random import randn
import math

from VCCell import VCCell

def crop(z, ri, ci, Sl):
	x = (math.ceil(ri-Sl[0]/2), math.ceil(ri+Sl[0]/2))
	y = (math.ceil(ci-Sl[1]/2), math.ceil(ci+Sl[1]/2))
	res = np.zeros(Sl)
	for i in range(*x):
		for j in range(*y):
			if (i >= 0) and (j >= 0) and (i < len(z)) and (j < len(z[0])):
				res[i+math.floor(Sl[0]/2)-ri][j+math.floor(Sl[1]/2)-ci] = z[i][j]
	return res

class VCPlane:
	def __init__(self, n_cl, Sl, c=None):
		self.n_cl = n_cl
		self.Sl = Sl
		self.c = c if c is not None else np.absolute(randn(*Sl))
		self.cells = [[VCCell(self.c) for _ in range(n_cl)] for _ in range(n_cl)]

	def feed_forward(self, u_cl):
		return np.array([[cell.feed_forward([crop(u_cl_i, ri, ci, self.Sl) for u_cl_i in u_cl]) \
			for ci, cell in enumerate(row)] for ri, row in enumerate(self.cells)])