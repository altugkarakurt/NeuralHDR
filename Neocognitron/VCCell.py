import numpy as np
import math

class VCCell:
	def __init__(self, c):
		self.c = c

	def feed_forward(self, u_cl):
		return math.sqrt(sum([np.dot(np.reshape(self.c, [-1]), np.reshape((u_cl_i ** 2), [-1])) for u_cl_i in u_cl]))
		#v_cl shape:1
		