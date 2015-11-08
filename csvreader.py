import matplotlib.pyplot as plt
import csv
import numpy as np
import sys
from random import randint
from zipfile import ZipFile
import io

with ZipFile('./Data/Digit.zip') as data_zip:
	data_test = data_zip.open('testDigit.csv')
	data_test = io.TextIOWrapper(data_test)
	reader = csv.reader(data_test)

	temp = np.array([[int(row[i]) for i in range(785)] for row in reader])
	images = [[row[l*28+1:(l+1)*28+1] for l in range(28)] for row in temp]
	labels = temp[:,0]
	
	for _ in range(5):
		idx = randint(0, 9000)
		print(labels[idx])
		plt.imshow(images[idx], cmap=plt.cm.gray_r, interpolation='nearest')
		plt.show()
