import numpy as np
from MLP import MLP

train_samples = [np.array([0,0]), np.array([0,1]), np.array([1,0]), np.array([1,1])]
train_labels = [[0], [1], [1], [0]]

mlp = MLP([2, 1])
train_errors = np.zeros(30)

for epoch_idx in range(30):
	mlp.train(train_samples, train_labels, epochs=100, block_size=1, learn_rate=0.5)
	print("Epoch %d done!" % (epoch_idx))
	print("%.2f %.2f %.2f %.2f" % (mlp([0,0]), mlp([0,1]), mlp([1,0]), mlp([1,1])))
	print()