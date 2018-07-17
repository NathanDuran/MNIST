import gzip
import csv
import pickle
import numpy as np

data_path = 'data/'

train_x = []
train_y = []
test_x = []
test_y = []

# Unzip csv file
with gzip.open(data_path + 'mnist_train.csv.gz', 'rt') as file:
    csv_data = csv.reader(file, delimiter=',')

    for line in csv_data:
        # First column are the labels
        train_y.append(line[0])
        # Remaining columns are the data
        train_x.append(line[1:])

with gzip.open(data_path + 'mnist_test.csv.gz', 'rt') as file:
    csv_data = csv.reader(file, delimiter=',')

    for line in csv_data:
        # First column are the labels
        test_y.append(line[0])
        # Remaining columns are the data
        test_x.append(line[1:])

# Convert to numpy arrays
train_y = np.array(train_y, dtype='int')
train_x = np.array(train_x, dtype='int')
test_x = np.array(test_x, dtype='int')
test_y = np.array(test_y, dtype='int')

# Calculate number of classes
num_classes = len(np.unique(train_y))

# Convert labels to one-hot encodings
train_y = np.eye(num_classes, dtype='int')[train_y]
test_y = np.eye(num_classes, dtype='int')[test_y]

# Pickle
with gzip.open(data_path + 'mnist.pkl', 'wb') as file:
    pickle.dump(train_x, file)
    pickle.dump(train_y, file)
    pickle.dump(test_x, file)
    pickle.dump(test_y, file)
