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
train_images = np.array(train_images, dtype='float32')
train_labels = np.array(train_labels, dtype='int32')
test_images = np.array(test_images, dtype='float32')
test_labels = np.array(test_labels, dtype='int32')

# Calculate number of classes
num_classes = len(np.unique(train_labels))

# Convert labels to one-hot encodings
train_labels = np.eye(num_classes, dtype='float32')[train_labels]
test_labels = np.eye(num_classes, dtype='float32')[test_labels]

# Pickle
data = {'train_images': train_images,
        'train_labels': train_labels,
        'test_images': test_images,
        'test_labels': test_labels}

with gzip.open(data_path + 'mnist.pkl', 'wb') as file:
    pickle.dump(data, file)

