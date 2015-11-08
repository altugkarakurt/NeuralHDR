import matplotlib.pyplot as plt
import csv
import numpy as np
import sys
from random import randint

f = open('./Data/testDigit.csv', 'rU')
reader = csv.reader(f)

temp = np.array([[int(row[i]) for i in range(785)] for row in reader])
images = [[row[l*28+1:(l+1)*28+1] for l in range(28)] for row in temp]
labels = temp[:,0]

for _ in range(5):
	idx = randint(0, 9000)
	print labels[idx] 
	plt.imshow(images[idx], cmap=plt.cm.gray_r, interpolation='nearest')
	plt.show()
