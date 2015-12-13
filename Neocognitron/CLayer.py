import numpy as np
from numpy.random import randn
import math

from CPlane import CPlane

class CLayer:
	def __init__(self, K_cl, n_cl, n_sl, Dl, d=None):
		self.n_cl = n_cl
		self.n_sl = n_sl
		self.K_cl = K_cl
		self.Dl = Dl

		self.d = d if d is not None else np.absolute(randn(*Dl))

		self.planes = [CPlane(n_cl, n_sl, Dl, d) for _ in range(K_cl)]

	def feed_forward(self, u_sl):
		return [plane.feed_forward(u_sl_i) for u_sl_i, plane in zip(u_sl, self.planes)]