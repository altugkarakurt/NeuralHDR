import matplotlib.pyplot as plt
import csv
import numpy as np
import sys
from random import randint
from zipfile import ZipFile
import io

class MNIST_Database:
	def __init__(self):
		with ZipFile('./Data/Digit.zip') as data_zip:
			data = data_zip.open('testDigit.csv')
			data = io.TextIOWrapper(data)
			reader = csv.reader(data)

			temp = np.array([[int(row[i]) for i in range(785)] for row in reader])
			self.test_images = [[row[l*28+1:(l+1)*28+1] for l in range(28)] for row in temp]
			self.test_labels = temp[:,0]
			data.close()
			
			data = data_zip.open('trainDigit.csv')
			data = io.TextIOWrapper(data)
			reader = csv.reader(data)
			
			temp = np.array([[int(row[i]) for i in range(785)] for row in reader])
			self.train_images = [[row[l*28+1:(l+1)*28+1] for l in range(28)] for row in temp]
			self.train_labels = temp[:,0]
			data.close()

sets = MNIST_Database()

for _ in range(5):
	idx = randint(0, 9000)
	print(sets.test_labels[idx])
	plt.imshow(sets.test_images[idx], cmap=plt.cm.gray_r, interpolation='nearest')
	plt.show()
