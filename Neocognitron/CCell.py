import numpy as np

def phi(x): return x if x > 0 else 0
def psi(x): return phi(x) / (0.5 + phi(x)) #alpha
 
class CCell:
	#def __init__(self, j, d):
	def __init__(self, d):
		#self.j = j
		self.d = d

	def feed_forward(self, u_sl, v_sl):
		#note: d becomes list when js are enabled
		#numerator = 1 + sum([np.dot(np.reshape(self.d, [-1]), np.reshape(u_sl[i], [-1])) * self.j[i] for i, _ in enumerate(u_sl)])
		numerator = 1 + np.dot(np.reshape(self.d, [-1]), np.reshape(u_sl, [-1]))
		denom = 1 + v_sl
		return psi(numerator/denom)
		#u_cl(k) shape:1