import matplotlib.pyplot as plt
import csv
import numpy as np
import sys
from random import randint
from zipfile import ZipFile
import io

def readData():
	with ZipFile('./Data/Digit.zip') as data_zip:
		data = data_zip.open('testDigit.csv')
		data = io.TextIOWrapper(data)
		reader = csv.reader(data)
	
		temp = np.array([[int(row[i]) for i in range(785)] for row in reader])
		test_images = [[row[l*28+1:(l+1)*28+1] for l in range(28)] for row in temp]
		test_labels = temp[:,0]
		data.close()
		
		data = data_zip.open('trainDigit.csv')
		data = io.TextIOWrapper(data)
		reader = csv.reader(data)
		
		temp = np.array([[int(row[i]) for i in range(785)] for row in reader])
		train_images = [[row[l*28+1:(l+1)*28+1] for l in range(28)] for row in temp]
		train_labels = temp[:,0]
		data.close()
		
	return (train_labels, train_images, test_labels, test_images)


class MNIST_Database:
	(train_labels, train_images, test_labels, test_images) = readData()
	
	@staticmethod
	def get(index, setname, flat=False, normalize=False, decode=False):
		img = MNIST_Database.image(index, setname, flat=flat, normalize=normalize)
		lbl = MNIST_Database.label(index, setname, decode=decode)
		return (lbl, img)
	
	@staticmethod
	def image(index, setname, flat=False, normalize=False):
		img = 0;
		if setname == 'test':		
			img = MNIST_Database.test_images[index]
		elif setname == 'train':
			img = MNIST_Database.train_images[index]
		
		if flat:
			img = np.reshape(img, -1)
		
		if normalize:
			img = img / np.max(img)
			
		return np.array(img)
		
	@staticmethod
	def label(index, setname, decode=False):
		lbl = -1;
		if setname == 'test':		
			lbl = MNIST_Database.test_labels[index]
		elif setname == 'train':
			lbl = MNIST_Database.train_labels[index]
		
		if decode:
			declbl = np.zeros((10))
			declbl[lbl] = 1
			lbl = declbl
			
		return np.array(lbl)

"""
sets = MNIST_Database()

for _ in range(5):
	idx = randint(0, 9000)
	print(sets.label(idx, 'train'))
	plt.imshow(sets.image(idx, 'train'), cmap=plt.cm.gray_r, interpolation='nearest')
	plt.show()
"""