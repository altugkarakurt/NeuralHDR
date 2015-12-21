import numpy as np
from numpy.random import randn
import math

from CCell import CCell

def crop(z, ri, ci, Dl):
	x = (math.ceil(ri-Dl[0]/2), math.ceil(ri+Dl[0]/2))
	y = (math.ceil(ci-Dl[1]/2), math.ceil(ci+Dl[1]/2))
	res = np.zeros(Dl)
	for i in range(*x):
		for j in range(*y):
			if (i >= 0) and (j >= 0) and (i < len(z)) and (j < len(z[0])):
				res[i+math.floor(Dl[0]/2)-ri][j+math.floor(Dl[1]/2)-ci] = z[i][j]
	return res

# def crop(z, ri, ci, Dl):
# 	x = (math.ceil(ri-Dl[0]/2), math.ceil(ri+Dl[0]/2)-1)
# 	y = (math.ceil(ci-Dl[1]/2), math.ceil(ci+Dl[1]/2)-1)
# 	vx = (max(x[0], 0), min(x[1], len(z)-1))
# 	px = (abs(vx[0] - x[0]), abs(vx[1] - x[1]))
# 	vy = (max(y[0], 0), min(y[1], len(z[0])-1))
# 	py = (abs(vy[0] - y[0]), abs(vy[1] - y[1]))
# 	return np.pad(z[vx[0]:vx[1]+1, vy[0]:vy[1]+1], (px, py), 'constant')

class CPlane:
	#def __init__(self, n_cl, n_sl, j=None, d=None):
	def __init__(self, n_cl, Dl, d=None):
		self.n_cl = n_cl
		self.Dl = Dl

		#self.j = j if j is not None else [np.absolute(randn()) for _ in range(K_sl)]
		self.d = d if d is not None else np.absolute(randn(*Dl))

		#self.cells = [[CCell(self.j, self.d) for _ in range(n_cl)] for _ in range(n_cl)]
		self.cells = [[CCell(self.d) for _ in range(n_cl)] for _ in range(n_cl)]

	def feed_forward(self, u_sl, v_sl):
		v_sl = crop(v_sl, int(np.shape(v_sl)[0]/2), int(np.shape(v_sl)[1]/2), np.shape(self.cells))
		return np.array([[cell.feed_forward(crop(u_sl, ri, ci, self.Dl), v_sl[ri][ci]) \
			for ci, cell in enumerate(row)] for ri, row in enumerate(self.cells)])
		#u_cl(k) shape: n*n