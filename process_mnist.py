import gzip
import csv
import pickle
import numpy as np

data_path = 'data/'

train_images = []
train_labels = []
test_images = []
test_labels = []

# Unzip csv file
with gzip.open(data_path + 'mnist_train.csv.gz', 'rt') as file:
    csv_data = csv.reader(file, delimiter=',')

    for line in csv_data:
        # First column are the labels
        train_labels.append(line[0])
        # Remaining columns are the data
        train_images.append(line[1:])

with gzip.open(data_path + 'mnist_test.csv.gz', 'rt') as file:
    csv_data = csv.reader(file, delimiter=',')

    for line in csv_data:
        # First column are the labels
        test_labels.append(line[0])
        # Remaining columns are the data
        test_images.append(line[1:])

# Convert to numpy arrays
train_images = np.array(train_images, dtype='int')
train_labels = np.array(train_labels, dtype='int')
test_images = np.array(test_images, dtype='int')
test_labels = np.array(test_labels, dtype='int')

# Calculate number of classes
# num_classes = len(np.unique(train_y))
#
# # Convert labels to one-hot encodings
# train_y = np.eye(num_classes, dtype='int')[train_y]
# test_y = np.eye(num_classes, dtype='int')[test_y]

# Pickle
data = {'train_images': train_images, 'train_labels': train_labels, 'test_images': test_images, 'test_labels': test_labels}
with gzip.open(data_path + 'mnist.pkl', 'wb') as file:
    pickle.dump(data, file)
    # pickle.dump(train_images, file)
    # pickle.dump(train_labels, file)
    # pickle.dump(test_images, file)
    # pickle.dump(test_labels, file)
