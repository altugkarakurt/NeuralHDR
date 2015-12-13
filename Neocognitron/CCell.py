import numpy as np

def phi(x): return x if x > 0 else 0
def psi(x): return phi(x) / (1 + phi(x))
 
class CCell:
	def __init__(self, d):
	#	self.j = j
		self.d = d

	def feed_forward(self, u_sl):
		#return psi(sum([np.dot(np.reshape(self.d, [-1]), np.reshape(u_sl[i], [-1])) * self.j[i] for i, _ in enumerate(u_sl)]))
		return psi(np.dot(np.reshape(self.d, [-1]), np.reshape(u_sl, [-1])))