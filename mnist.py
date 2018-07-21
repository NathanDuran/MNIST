import os
import gzip
import pickle
from math import trunc

import utilities as utils
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow debugging
########
# https://www.oreilly.com/learning/not-another-mnist-tutorial-with-tensorflow

# Load training and test data
with gzip.open('data/mnist.pkl', 'rb') as file:
    data = pickle.load(file)

train_images = data['train_images']
train_labels = data['train_labels']
test_images = data['test_images']
test_labels = data['test_labels']

# Data parameters
input_size = train_images.shape[1]  # 784 (28 x 28)
num_classes = len(np.unique(train_labels))  # 10
num_training_examples = train_images.shape[0]  # 60,000
num_test_examples = test_images.shape[0]  # 10,000

print("Input size: ", input_size)
print("Number of classes: ", num_classes)
print("Number of training examples:", num_training_examples)
print("Number of test examples:", num_test_examples)

# Network parameters
num_hidden_nodes = 500

# Training parameters
batch_size = 100
num_epochs = 10
learning_rate = 0.01

# Display a random digit
# utils.display_digit(train_x, train_y, np.random.randint(0, train_x.shape[0]))

# Image and label placeholder variables
images_placeholder = tf.placeholder(tf.float32, [None, input_size])
labels_placeholder = tf.placeholder(tf.int32)

# Make Tensorflow dataset and iterator
dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder)).batch(batch_size).repeat(1)
iterator = dataset.make_initializable_iterator()
images, labels = iterator.get_next()


def neural_network_model(input_data):
    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([input_size, num_hidden_nodes])),
                      'biases': tf.Variable(tf.random_normal([num_hidden_nodes]))}

    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([num_hidden_nodes, num_hidden_nodes])),
                      'biases': tf.Variable(tf.random_normal([num_hidden_nodes]))}

    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([num_hidden_nodes, num_hidden_nodes])),
                      'biases': tf.Variable(tf.random_normal([num_hidden_nodes]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([num_hidden_nodes, num_classes])),
                    'biases': tf.Variable(tf.random_normal([num_classes]))}

    l1 = tf.add(tf.matmul(input_data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


# Pass data through the model
prediction = neural_network_model(images)

# Calculate the cost
cost = tf.reduce_sum(tf.losses.sparse_softmax_cross_entropy(labels, prediction))

# Minimise the cost with optimisation function
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Count how many predictions (index of highest probability) match the labels
correct = tf.equal(tf.argmax(prediction, axis=1), tf.cast(labels, tf.int64))

# Count how many are correct
num_correct = tf.reduce_sum(tf.cast(correct, tf.int32))

# Calculate the average of correct predictions
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.Session() as sess:

    # Initialise all the variables
    sess.run(tf.global_variables_initializer())

    # Loop for number of training epochs
    for epoch in range(1, num_epochs + 1):
        epoch_loss = batch_loss = 0
        epoch_accuracy = batch_accuracy = 0

        # Initialise the iterator with the training data
        sess.run(iterator.initializer, feed_dict={images_placeholder: train_images, labels_placeholder: train_labels})

        # Loop over each batch once per epoch
        while True:
            try:
                _, batch_loss, batch_accuracy = sess.run([optimizer, cost, accuracy])
            except tf.errors.OutOfRangeError:
                break

        epoch_loss += batch_loss
        epoch_accuracy += batch_accuracy
        print("Epoch: {}/{}, loss: {:.3f}, accuracy: {:.2f}%".format(epoch, num_epochs, epoch_loss, epoch_accuracy * 100))

    test_epochs = 20
    average_accuracy = total_num_correct = 0
    # Loop over each batch once per epoch
    for epoch in range(1, test_epochs + 1):
        epoch_accuracy = batch_accuracy = 0
        batch_num_correct = 0

        # Initialise the iterator with the test data
        sess.run(iterator.initializer, feed_dict={images_placeholder: test_images, labels_placeholder: test_labels})
        while True:
            try:
                batch_accuracy, batch_num_correct = sess.run([accuracy, num_correct])
                total_num_correct += batch_num_correct
            except tf.errors.OutOfRangeError:
                break
        epoch_accuracy += batch_accuracy
        average_accuracy += batch_accuracy
        print("Epoch: {}/{}, accuracy: {:.2f}%".format(epoch, test_epochs, epoch_accuracy * 100))

    print("Average test set accuracy over {} iterations is {:.2f}%".format(test_epochs, (average_accuracy * 100) / test_epochs))
    print("Number Correct {:.0f}/{}".format((total_num_correct / test_epochs), num_test_examples))

    # Run on whole test set once
    print("Accuracy: {:.2f}%".format(accuracy.eval(feed_dict={images: test_images, labels: test_labels}) * 100))
    print("Number Correct: {}/{}".format((num_correct.eval(feed_dict={images: test_images, labels: test_labels})), num_test_examples))