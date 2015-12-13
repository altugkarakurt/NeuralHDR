import numpy as np
from numpy.random import randn
import math

from SPlane import SPlane
from VPlane import VPlane

class SLayer:
	def __init__(self, K_cl, K_sl, n_cl, n_sl, Al, a=None, b=None, c=None, r=None, sigma=None):
		self.n_cl = n_cl
		self.n_sl = n_sl
		self.K_cl = K_cl
		self.K_sl = K_sl
		self.Al = Al

		self.a = a if a is not None else [[np.absolute(randn(*Al)) for _ in range(K_cl)] for _ in range(K_sl)]
		self.b = b if b is not None else [np.absolute(randn()) for _ in range(K_sl)]
		self.c = c if c is not None else np.absolute(randn(*Al))
		self.r = r if r is not None else np.absolute(randn())
		self.sigma = sigma if sigma is not None else np.absolute(randn())

		self.planes = [SPlane(K_cl, n_cl, n_sl, Al, self.a[k], self.b[k], self.r, self.sigma) for k in range(K_sl)]
		self.vplane = VPlane(K_cl, n_cl, n_sl, Al, c)

	def feed_forward(self, u_cl):
		u_vl = self.vplane.feed_forward(u_cl)
		return [plane.feed_forward(u_cl, u_vl) for plane in self.planes]
