import numpy as np
from numpy.random import randn
import math

from SPlane import SPlane
from VSPlane import VSPlane

class SLayer:
	def __init__(self, K_cl, K_sl, n_cl, n_sl, Sl, Dl, a=None, b=None, d=None, r=None):
		self.K_cl = K_cl
		self.K_sl = K_sl
		self.n_cl = n_cl
		self.n_sl = n_sl
		self.Sl = Sl
		self.Dl = Dl

		self.a = a if a is not None else [[np.absolute(randn(*Sl)) for _ in range(K_cl)] for _ in range(K_sl)]
		self.b = b if b is not None else [np.absolute(randn()) for _ in range(K_sl)]
		if d is None: raise ValueError('d cannot be None')
		self.d = d
		self.r = r if r is not None else np.absolute(randn())

		self.planes = [SPlane(K_cl, n_sl, Sl, self.a[k], self.b[k], self.r) for k in range(K_sl)]
		self.vplane = VSPlane(n_sl, Dl, d)

	def feed_forward(self, u_cl, v_cl):
		u_sl = [plane.feed_forward(u_cl, v_cl) for plane in self.planes]
		v_sl = self.vplane.feed_forward(u_sl)
		return (u_sl, v_sl)
