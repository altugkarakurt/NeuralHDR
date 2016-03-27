import numpy as np
import time
import sys

from MNIST_Database import MNIST
from MLP import MLP

# class Logger(object):
	# def __init__(self):
		# self.terminal = sys.stdout
		# self.log = open(time.strftime("./Logs/%Y%m%d-%H%M%S-logfile.log"), "a")		

	# def write(self, message):
		# self.terminal.write(message)
		# self.log.write(message)  

	# def flush(self):
		# self.log.flush()
		# self.terminal.flush()
		# pass

# sys.stdout = Logger()

demo_count = 1000
demo_samples = MNIST.get(*range(demo_count), setname='demo', decode=True, flat=True, normalize=True)

mlp = MLP(784, [30, 10], weights='20151221-062115-weights-ep10-0.9290-0.9284')

if demo_samples[315]['label'] is None:
	print('ok. no labels.')

with open('mpllables.txt', 'wt') as lblfile:
	for sample in demo_samples:
		label = np.argmax(mlp(sample['image']))
		print(label)
		print(label, file=lblfile)
		sample['label'] = label
		
# sys.stdout.flush()
# sys.stdout = sys.stdout.terminal
