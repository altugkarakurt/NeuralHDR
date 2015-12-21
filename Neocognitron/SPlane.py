import numpy as np
from numpy.random import randn
import math

from SCell import SCell

# def crop(z, ri, ci, Sl):
# 	x = (math.ceil(ri-Sl[0]/2), math.ceil(ri+Sl[0]/2))
# 	y = (math.ceil(ci-Sl[1]/2), math.ceil(ci+Sl[1]/2))
# 	res = np.zeros(Sl)
# 	for i in range(*x):
# 		for j in range(*y):
# 			if (i >= 0) and (j >= 0) and (i < len(z)) and (j < len(z[0])):
# 				res[i+math.floor(Sl[0]/2)-ri][j+math.floor(Sl[1]/2)-ci] = z[i][j]
# 	return res

def crop(z, ri, ci, Dl):
	x = (math.ceil(ri-Dl[0]/2), math.ceil(ri+Dl[0]/2)-1)
	y = (math.ceil(ci-Dl[1]/2), math.ceil(ci+Dl[1]/2)-1)
	vx = (max(x[0], 0), min(x[1], len(z)-1))
	px = (abs(vx[0] - x[0]), abs(vx[1] - x[1]))
	vy = (max(y[0], 0), min(y[1], len(z[0])-1))
	py = (abs(vy[0] - y[0]), abs(vy[1] - y[1]))
	return np.pad(z[vx[0]:vx[1]+1, vy[0]:vy[1]+1], (px, py), 'constant')

class SPlane:
	def __init__(self, K_cl, n_sl, Sl, a=None, b=None, r=None):
		self.n_sl = n_sl
		self.K_cl = K_cl
		self.Sl = Sl

		self.a = a if a is not None else [np.absolute(randn(*Sl)) for _ in range(K_cl)]
		self.b = b if b is not None else np.absolute(randn())
		self.r = r if r is not None else np.absolute(randn())

		self.cells = [[SCell(a, b, r) for _ in range(n_sl)] for _ in range(n_sl)]

	def feed_forward(self, u_cl, v_cl):
		v_cl = crop(v_cl, int(np.shape(v_cl)[0]/2), int(np.shape(v_cl)[1]/2), np.shape(self.cells))
		return np.array([[cell.feed_forward([crop(u_cl_i, ri, ci, self.Sl) for u_cl_i in u_cl], v_cl[ri][ci]) \
			for ci, cell in enumerate(row)] for ri, row in enumerate(self.cells)])
		#u_sl(k) shape n*n