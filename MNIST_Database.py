import csv
import numpy as np
from zipfile import ZipFile
import io
import os
from copy import deepcopy
from PIL import Image

def read_data_from_zip(csv_filename):
	with ZipFile('./Data/Digit.zip') as data_zip:
		data_f = data_zip.open(csv_filename)
		data = io.TextIOWrapper(data_f)
		
		reader = csv.reader(data)
		temp = [{'label':int(row[0]), 'image':np.reshape([int(pixel) for pixel in row[1:]], [28, 28])} for row in reader]
		
		data.close()
		data_f.close()
	return temp
	

def read_data_from_csv(csv_filename, labeled=True):
	with open('./Data/' + csv_filename, 'rt') as data:
		reader = csv.reader(data)
		if labeled:
			temp = [{'label':int(row[0]), 'image':np.reshape([int(pixel) for pixel in row[1:]], [28, 28])} for row in reader]
		else:
			temp = [{'label':None, 'image':np.reshape([int(pixel) for pixel in row], [28, 28])} for row in reader]
		
	return temp
	

def read_data(setname):
	npz_directory = './Data/npz/'
	npz_filename = npz_directory + setname + '.npz'
	
	if not os.path.exists(npz_directory):
		os.makedirs(npz_directory)
	
	try:
		if os.path.exists(npz_filename):
			with open(npz_filename, 'rb') as npz_file:
				npz_vars = np.load(npz_file)
				return npz_vars['dataset'].tolist()
	except RuntimeError:
		os.remove(npz_filename)
	
	csvfiles = {'test':'testDigit.csv', 'train':'trainDigit.csv', 'demo':'demoDataset.csv'}
	
	if setname in ['test', 'train']:
		data = read_data_from_zip(csvfiles[setname])
	elif setname == 'demo':
		data = read_data_from_csv(csvfiles[setname], labeled=False)
	elif setname in MNIST.sets:
		raise NotImplementedError('Set {} not implemented yet'.format(setname))
	else:
		raise ValueError('Invalid set name')
	
	with open(npz_filename, 'wb') as npz_file:
		np.savez(npz_file, dataset=data)
	return data
	

class MNIST:
	sets = ['train', 'test', 'demo']
	data = {}
	for setname in sets:
		data[setname] = read_data(setname)
	
	@staticmethod
	def flatten_image(img):
		return img.flatten()
	
	@staticmethod
	def normalize_image(img, scale=(0, 1)):
		return (img - np.min(img)) / (np.max(img) - np.min(img)) \
			* (np.max(scale) - np.min(scale)) + np.min(scale)
	
	@staticmethod
	def resize_image(img, scale=(28, 28)):
		return np.array(Image.frombytes('L', (28,28), np.uint8(img)).resize(scale))
	
	@staticmethod
	def decode_label(lbl):
		decoded = np.zeros([10])
		decoded[lbl] = 1
		return decoded
	
	@staticmethod
	def get(*indices, setname='train', flat=False, normalize=False, decode=False, resize=False):
		if setname not in MNIST.sets:
			raise ValueError('Invalid set name')
		
		samples = [deepcopy(MNIST.data[setname][index]) for index in indices]
		
		if flat or normalize or decode or resize:
			for sample in samples:
				if resize:
					sample['image'] = MNIST.resize_image(sample['image'], scale=resize)
				if flat:
					sample['image'] = MNIST.flatten_image(sample['image'])
				if normalize:
					sample['image'] = MNIST.normalize_image(sample['image'])
				if decode:
					sample['label'] = MNIST.decode_label(sample['label'])
		
		if len(samples) == 1:
			return samples[0]
		return samples
	
	@staticmethod
	def image(*index, **kwargs):
		samples = MNIST.get(*index, **kwargs)
		
		if type(samples) is list:
			return [sample['image'] for sample in samples]
		return samples['image']
		
	@staticmethod
	def label(*index, **kwargs):
		samples = MNIST.get(*index, **kwargs)
		
		if type(samples) is list:
			return [sample['label'] for sample in samples]
		return samples['label']
	

