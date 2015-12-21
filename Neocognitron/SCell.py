import numpy as np

def phi(x): return x if x > 0 else 0

class SCell:
	def __init__(self, a, b, r):
		self.r = r
		self.a = a
		self.b = b

	def feed_forward(self, u_cl, v_cl):
		numerator = 1 + np.dot(np.reshape(self.a, [-1]), np.reshape(u_cl, [-1]))
		denom = 1 + (2*self.r/(1+self.r) * self.b * v_cl)
		return phi((numerator/denom) - 1) * self.r
		#u_sl(k) shape:1


		