import numpy as np
from numpy.random import randn
from Neocognitron import Neocognitron

image = np.absolute(randn(19,19))
n = Neocognitron(19, [(19,12), (21,8), (21,38), (13,19), (13,35), (7,23), (3,11), (1,10)])
n.feed_forward(image)
