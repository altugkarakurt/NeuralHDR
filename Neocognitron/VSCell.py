import numpy as np
import math

class VSCell:
	def __init__(self, d):
		self.d = d

	def feed_forward(self, u_sl):
		return np.mean([np.dot(np.reshape(self.d, [-1]), np.reshape(u_sl_i, [-1])) for u_sl_i in u_sl], axis=0)
		#v_sl shape:n*n