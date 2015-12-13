import numpy as np

def phi(x): return x if x > 0 else 0

class SCell:
	def __init__(self, a, b, r, sigma):
		self.r = r
		self.a = a
		self.b = b
		self.sigma = sigma

	def feed_forward(self, u_cl, u_vl):
		numerator = self.sigma + np.dot(np.reshape(self.a, [-1]), np.reshape(u_cl, [-1]))
		denom = self.sigma + (self.r/(1+self.r) * self.b * u_vl)
		return phi((numerator/denom) - 1) * self.r


		