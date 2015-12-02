from MNIST_Database import MNIST_Database as MNIST
import numpy as np
from MLP import MLP

train_labels = MNIST.label(*range(60000), setname='train', decode=True)
train_samples = MNIST.image(*range(60000), setname='train', flat=True, normalize=True)
test_labels = MNIST.label(*range(9000), setname='test', decode=True)
test_samples = MNIST.image(*range(9000), setname='test', flat=True, normalize=True)

mlp = MLP([784, 30, 10])
train_errors = np.zeros(30)
test_errors = np.zeros(30)

for epoch_idx in range(30):
	mlp.train(train_samples, train_labels, epochs=1, block_size=10, learn_rate=3.0)
	train_errors[epoch_idx] = mlp.validate(train_samples, train_labels) #training error
	test_errors[epoch_idx] = mlp.validate(test_samples, test_labels) #test error
	print("Epoch %d done!" % (epoch_idx))
	print("Training Accuracy: %.2f" % (train_errors[epoch_idx]))
	print("Test Accuracy: %.2f" % (test_errors[epoch_idx]))
	print("------------------------------------")
	mlp.save_weights("weights-%.2f-%.2f" % (train_errors[epoch_idx], test_errors[epoch_idx]))