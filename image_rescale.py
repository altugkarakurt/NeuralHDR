from PIL import Image
import numpy as np
from MNIST_Database import MNIST

"""--------------------------------------------------------------------
This script scales the 28x28 images in MNIST database to 19x19 to be
directly used in Fukushima's original Neocognitron configuration.
--------------------------------------------------------------------"""
# Rescale training set
temp = np.array([MNIST.data["train"][i]["image"] for i in range(60000)])
rescaled = []
for t in temp:
	img = Image.frombytes('L', (28,28), np.uint8(t))
	rescaled.append(np.array(img.resize((19,19))))

rescaled = np.array(rescaled)

npz_path = './Data/npz/neocognitron/train.npz'
with open(npz_path, 'wb') as npz_file:
		np.savez(npz_file, dataset=rescaled)

# Rescale test set
temp = np.array([MNIST.data["test"][i]["image"] for i in range(9000)])
rescaled = []
for t in temp:
	img = Image.frombytes('L', (28,28), np.uint8(t))
	rescaled.append(np.array(img.resize((19,19))))

rescaled = np.array(rescaled)

npz_path = './Data/npz/neocognitron/test.npz'
with open(npz_path, 'wb') as npz_file:
		np.savez(npz_file, dataset=rescaled)