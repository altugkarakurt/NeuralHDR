from MNIST_Database import MNIST_Database as MNIST
import numpy as np
from MLP import MLP

mlp = MLP([784, 30, 10])
labels = [MNIST.label(i, setname='train', decode=True) for i in range(60000)]
samples = [MNIST.image(i, setname='train', flat=True, normalize=True)[0] for i in range(60000)]
mlp.train(samples, labels, epochs=30, block_size=10, learn_rate=3.0)