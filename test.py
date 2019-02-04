# Data and Tensorboard path
import datetime
import gzip
import pickle
import time
from mnist_utilities import process_mnist, display_digit

data_path = 'data/mnist.pkl'
tensorboard_path = 'output/multi_layer/tb_logs'
image_path = 'output/multi_layer/images'

# start_time = time.time()
# print("Training started: " + datetime.datetime.now().strftime("%b %d %T"))
#
# # Load training and test data
# with gzip.open(data_path, 'rb') as file:
#     data = pickle.load(file)
# print("Loaded data from file %s." % data_path)
#
# train_images = data['train_images']
# train_labels = data['train_labels']
# test_images = data['test_images']
# test_labels = data['test_labels']
#
# end_time = time.time()
# print("Training took " + str(('%.3f' % (end_time - start_time))))

start_time = time.time()
print("Training started: " + datetime.datetime.now().strftime("%b %d %T"))

# Load training and test data
train_images, train_labels = process_mnist('data/mnist.zip', 'train')

test_images, test_labels = process_mnist('data/mnist.zip', 'test')


end_time = time.time()
print("Training took " + str(('%.3f' % (end_time - start_time))))

print(train_images.shape)
print(train_labels.shape)
print(type(train_images))
print(type(train_labels))
display_digit(train_images, train_labels, 4)