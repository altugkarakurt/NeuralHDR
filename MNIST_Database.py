import csv
import numpy as np
from zipfile import ZipFile
import io
import os.path

def read_data_from_zip(csv_filename):
	with ZipFile('./Data/Digit.zip') as data_zip:
		data = data_zip.open(csv_filename)
		data = io.TextIOWrapper(data)
		reader = csv.reader(data)
	
		temp = np.array([[int(row[i]) for i in range(785)] for row in reader])
		images = [[row[l*28+1:(l+1)*28+1] for l in range(28)] for row in temp]
		labels = temp[:,0]
		data.close()
	return (images, labels)


def read_data(setname):
	npz_filename = './Data/npz/' + setname + '.npz'
	if os.path.exists(npz_filename):
		with open(npz_filename, 'rb') as npz_file:
			npz_vars = np.load(npz_file)
			return (npz_vars['images'], npz_vars['labels'])
	
	if setname == 'test':
		(images, labels) = read_data_from_zip('testDigit.csv')
	elif setname == 'train':
		(images, labels) = read_data_from_zip('trainDigit.csv')
	else:
		raise ValueError('Not a valid set name')
	
	with open(npz_filename, 'wb') as npz_file:
		np.savez(npz_file, images=images, labels=labels)
	return (images, labels)
	

class MNIST_Database:
	(train_images, train_labels) = read_data('train')
	(test_images, test_labels) = read_data('test')
	
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
			raise ValueError('Not a valid set name')
			
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
			raise ValueError('Not a valid set name')
		
		if decode:
			declbl = np.zeros((10))
			declbl[lbl] = 1
			lbl = declbl
			
		return np.array(lbl)