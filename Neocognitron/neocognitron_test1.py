import numpy as np
from numpy.random import randn
from Neocognitron import Neocognitron

def windower(pl_size, l_size):
	m = abs(pl_size-l_size)+1
	return (m, m)

layer_sizes = [(19, 12), (21, 8), (21, 38), (13, 19), (13, 35), (7, 23), (3, 11), (1, 10)]
windows = [(3, 3), windower(19, 21), (9, 9), windower(21, 13), (19, 19), windower(13, 7), (19, 19), windower(7, 3), (1, 1)]
n = Neocognitron(19, layer_sizes, windows)

image = np.absolute(randn(19, 19))
a = n.estimate(image)
print(a, end= '\n\n')