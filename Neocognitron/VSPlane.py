import numpy as np
from numpy.random import randn
import math

from VSCell import VSCell

def crop(z, ri, ci, Dl):
	x = (math.ceil(ri-Dl[0]/2), math.ceil(ri+Dl[0]/2))
	y = (math.ceil(ci-Dl[1]/2), math.ceil(ci+Dl[1]/2))
	res = np.zeros(Dl)
	for i in range(*x):
		for j in range(*y):
			if (i >= 0) and (j >= 0) and (i < len(z)) and (j < len(z[0])):
				res[i+math.floor(Dl[0]/2)-ri][j+math.floor(Dl[1]/2)-ci] = z[i][j]
	return res

class VSPlane:
	def __init__(self, n_sl, Dl, d=None):
		self.n_sl = n_sl
		self.Dl = Dl
		self.d = d if d is not None else np.absolute(randn(*Dl))
		self.cells = [[VSCell(self.d) for _ in range(n_sl)] for _ in range(n_sl)]

	def feed_forward(self, u_sl):
		return np.array([[cell.feed_forward([crop(u_sl_i, ri, ci, self.Dl) for u_sl_i in u_sl]) \
			for ci, cell in enumerate(row)] for ri, row in enumerate(self.cells)])