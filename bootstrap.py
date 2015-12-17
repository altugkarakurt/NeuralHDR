# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import randint
from MNIST_Database import MNIST 

bins = [[],[],[],[],[],[],[],[],[],[]]

for sample in MNIST.data['train']:
	bins[sample['label']].append(sample["image"])

#bins[label][image#][28][28]

def bootstrap():
	lbl = randint(0,10)
	cnt = np.shape(bins[lbl])[0]
	bot = np.reshape([randint(0,cnt) for _ in range(784)], [28, 28])
	img = np.array([[bins[lbl][bot[i][j]][j][i] for i in range(28)] for j in range(28)])
	return {'label':lbl, 'image':img}