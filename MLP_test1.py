from MNIST_Database import MNIST_Database as MNIST
import numpy as np
from MLP import MLP
import time

train_count = 60000
train_labels = MNIST.label(*range(train_count), setname='train', decode=True)
train_samples = MNIST.image(*range(train_count), setname='train', flat=True, normalize=True)

test_count = 9000
test_labels = MNIST.label(*range(test_count), setname='test', decode=True)
test_samples = MNIST.image(*range(test_count), setname='test', flat=True, normalize=True)

mlp = MLP([784, 30, 10])

epoch_count = 10
train_errors = np.zeros(epoch_count)
test_errors = np.zeros(epoch_count)

print(time.strftime("Began training at %Y%d%m-%H%M%S"))
print()

for epoch_idx in range(1, epoch_count):
	mlp.train(train_samples, train_labels, epochs=1, block_size=1, learn_rate=1/epoch_idx)
	train_errors[epoch_idx-1] = mlp.validate(train_samples, train_labels) #training error
	test_errors[epoch_idx-1] = mlp.validate(test_samples, test_labels) #test error
	print("Epoch %d done!" % (epoch_idx))
	print("Training Accuracy: %.2f" % (train_errors[epoch_idx-1]))
	print("Test Accuracy: %.2f" % (test_errors[epoch_idx-1]))
	print("")
	mlp.save_weights("weights-ep%d-%.2f-%.2f" % (epoch_idx, train_errors[epoch_idx-1], test_errors[epoch_idx-1]))
	
print(time.strftime("Ended training at %Y%d%m-%H%M%S"))