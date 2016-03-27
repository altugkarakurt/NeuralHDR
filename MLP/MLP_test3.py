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

train_count = 60000
bootstrap_count = 10000
train_labels = MNIST.label(*range(train_count), setname='train', decode=True)
train_samples = MNIST.image(*range(train_count), setname='train', flat=True, normalize=True)

test_count = 9000
test_labels = MNIST.label(*range(test_count), setname='test', decode=True)
test_samples = MNIST.image(*range(test_count), setname='test', flat=True, normalize=True)

mlp = MLP(784, [30, 10])

epoch_count = 60
train_errors = np.zeros(epoch_count)
test_errors = np.zeros(epoch_count)

begin_time = time.time()
print(time.strftime("Began training at %Y%m%d-%H%M%S"))
print()

for epoch_idx in range(1, epoch_count+1):
	bootstrap_data = [MNIST.bootstrap() for _ in range(bootstrap_count)]
	bootstrap_labels = [MNIST.decode_label(sample['label']) for sample in bootstrap_data]
	bootstrap_samples = [MNIST.flatten_image(MNIST.normalize_image(sample['image'])) for sample in bootstrap_data]
	
	mlp.train(train_samples+bootstrap_samples, train_labels+bootstrap_labels, epochs=1, block_size=1, learn_rate=2/epoch_idx)
	train_errors[epoch_idx-1] = mlp.validate(bootstrap_samples, bootstrap_labels) #training error
	test_errors[epoch_idx-1] = mlp.validate(test_samples, test_labels) #test error
	print("Bs10k Epoch %d done at " % (epoch_idx) + time.strftime("%Y%m%d-%H%M%S"))
	print("Bs10k Training Accuracy: %.4f" % (train_errors[epoch_idx-1]))
	print("Bs10k Test Accuracy: %.4f" % (test_errors[epoch_idx-1]))
	print("")
	mlp.save_weights("weights-bs10k-ep%d-%.4f-%.4f" % (epoch_idx, train_errors[epoch_idx-1], test_errors[epoch_idx-1]))

end_time = time.time()
print(time.strftime("Bs10k Ended training at %Y%m%d-%H%M%S"))

delta = end_time - begin_time
print("Overall, Bs10k lasted %d hours %d minutes %d seconds" % (delta/3600, (delta/60)%60, delta%60));

# sys.stdout.flush()
# sys.stdout = sys.stdout.terminal
