import gzip
import pickle
import utilities as utils
import numpy as np
import tensorflow as tf

data_path = 'data/'

# Load training and test data
with gzip.open(data_path + 'mnist.pkl', 'rb') as file:
    train_x = pickle.load(file)
    train_y = pickle.load(file)
    test_x = pickle.load(file)
    test_y = pickle.load(file)

num_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
num_classes = 10
batch_size = 100

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
utils.display_digit(train_x, train_y, 1)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32)

# def neural_network_model(input_data):
#
#     hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
#                       'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
#
#     hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
#                       'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
#
#     hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
#                       'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
#
#     output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
#                     'biases': tf.Variable(tf.random_normal([n_classes]))}
#
#     l1 = tf.add(tf.matmul(input_data, hidden_layer_1['weights']), hidden_layer_1['biases'])
#     l1 = tf.nn.relu(l1)
#
#     l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
#     l2 = tf.nn.relu(l2)
#
#     l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
#     l3 = tf.nn.relu(l3)
#
#     output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
#
#     return output