from MNIST_Database import MNIST_Database as MNIST
import numpy as np
from MLP import MLP

train_labels = [MNIST.label(i, setname='train', decode=True) for i in range(60000)]
train_samples = [MNIST.image(i, setname='train', flat=True, normalize=True)[0] for i in range(60000)]
test_labels = [MNIST.label(i, setname='test', decode=True) for i in range(9000)]
test_samples = [MNIST.image(i, setname='test', flat=True, normalize=True)[0] for i in range(9000)]

mlp = MLP([784, 30, 10])
train_errors = np.zeros(30)
test_errors = np.zeros(30)

for epoch_idx in range(30):
	mlp.train(train_samples, train_labels, epochs=1, block_size=10, learn_rate=3.0, savename=("epoch" + str(epoch_idx) + ".npy"))
	train_errors[epoch_idx] = mlp.validate(train_samples, train_labels) #training error
	test_errors[epoch_idx] = mlp.validate(test_samples, test_labels) #test error
	print("Epoch %d done!" % (epoch_idx))
	print("Training Accuracy: %.2f" % (train_errors[epoch_idx]))
	print("Test Accuracy: %.2f" % (test_errors[epoch_idx]))
	print("------------------------------------")