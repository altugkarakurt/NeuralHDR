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
	def get(*index, setname=None, flat=False, normalize=False, decode=False):
		img = MNIST_Database.image(*index, setname=setname, flat=flat, normalize=normalize)
		lbl = MNIST_Database.label(*index, setname=setname, decode=decode)
		return (lbl, img)
	
	@staticmethod
	def image(*index, setname=None, flat=False, normalize=False):
		if setname == 'test':		
			img = [MNIST_Database.test_images[ind] for ind in index]
		elif setname == 'train':
			img = [MNIST_Database.train_images[ind] for ind in index]
		else:
			img = None
		if flat:
			img = np.reshape(img, [len(index), -1])
		
		if normalize:
			img = img / np.max(img)
			
		return np.array(img)
		
	@staticmethod
	def label(*index, setname=None, decode=False):
		if setname == 'test':
			lbl = [MNIST_Database.test_labels[ind] for ind in index]
		elif setname == 'train':
			lbl = [MNIST_Database.train_labels[ind] for ind in index]
		else:
			lbl = None
		
		if decode:
			declbl = np.zeros((10))
			declbl[lbl] = 1
			lbl = declbl
			
		return np.array(lbl)