from MNIST_Database import MNIST_Database as MNIST
import numpy as np
from MLP import MLP

mlp = MLP([784, 30, 10])
labels = [MNIST.label(i, setname='train', decode=True) for i in range(9000)]
samples = [MNIST.image(i, setname='train', flat=True, normalize=True) for i in range(9000)]
mlp.train(labels, samples, epochs=30, m=10, eta=3.0)