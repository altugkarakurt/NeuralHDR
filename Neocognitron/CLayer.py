import numpy as np
from numpy.random import randn
import math

from CPlane import CPlane
from VCPlane import VCPlane

class CLayer:
	def __init__(self, K_cl, n_cl, Dl, Sl, c=None, d=None):
		self.K_cl = K_cl
		self.n_cl = n_cl
		self.Dl = Dl
		self.Sl = Sl

		self.c = c if c is not None else np.absolute(randn(*Sl))
		if d is None: raise ValueError('d cannot be None')
		self.d = d

		self.planes = [CPlane(n_cl, Dl, d) for _ in range(K_cl)]
		self.vplane = VCPlane(n_cl, Sl, c)

	def feed_forward(self, u_sl, v_sl):
		u_cl = [plane.feed_forward(u_sl_i, v_sl) for u_sl_i, plane in zip(u_sl, self.planes)]
		v_cl = self.vplane.feed_forward(u_cl)
		return (u_cl, v_cl)