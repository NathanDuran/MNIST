import os
import time
import datetime
import gzip
import pickle
import numpy as np
import tensorflow as tf
from nn_utilities import nn_layer, make_dataset_placeholder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow debugging

# Data and Tensorboard path
data_path = 'data/mnist.pkl'
tensorboard_path = 'output'

# Load training and test data
with gzip.open(data_path, 'rb') as file:
    data = pickle.load(file)
print("Loaded data from file %s." % data_path)

train_images = data['train_images']
train_labels = data['train_labels']
test_images = data['test_images']
test_labels = data['test_labels']

# Data parameters
input_size = train_images.shape[1]  # 784 (28 x 28)
num_classes = len(np.unique(train_labels))  # 10
num_training_examples = train_images.shape[0]  # 60,000
num_test_examples = test_images.shape[0]  # 10,000

# Network parameters
num_hidden_nodes = 196

# Training parameters
batch_size = 100
learning_rate = 0.01
num_epochs = 10

print("------------------------------------")
print("Using parameters...")
print("Input size: ", input_size)
print("Number of classes: ", num_classes)
print("Number of training examples:", num_training_examples)
print("Number of test examples:", num_test_examples)
print("Batch size: ", batch_size)
print("Learning Rate: ", learning_rate)
print("Epochs: ", num_epochs)

# Display a random digit
# utils.display_digit(train_x, train_y, np.random.randint(0, train_x.shape[0]))

# Define Tensorflow Graph
print("------------------------------------")
print('Define Graph...')

# Define input dataset
iterator, images_placeholder, labels_placeholder = make_dataset_placeholder(input_size, batch_size, 'input_dataset')
images, labels = iterator.get_next()

# Define the network
hidden_1, _ = nn_layer(images, input_size, num_hidden_nodes, layer_name='hidden_layer_1')
hidden_2, _ = nn_layer(hidden_1, num_hidden_nodes, num_hidden_nodes, layer_name='hidden_layer_2')
hidden_3, hidden_weights = nn_layer(hidden_2, num_hidden_nodes, num_hidden_nodes, layer_name='hidden_layer_3')
prediction, _ = nn_layer(hidden_3, num_hidden_nodes, num_classes, activation_func=tf.identity, layer_name='output_layer')

# Calculate the cost
with tf.name_scope('cross_entropy'):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.to_int64(labels), logits=prediction)
    tf.summary.scalar('Loss', loss)

# Minimise the cost with optimisation function
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Calculate the average of correct predictions
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(prediction, axis=1), tf.to_int64(labels))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('Accuracy', accuracy)

# with tf.name_scope('tensorboard_input'):
#   tb_images = tf.placeholder(tf.float32, [None, input_size], name='tb_images')

# weight_summary = tf.summary.image('hidden_weights', tf.reshape(hidden_weights(images), [-1, 28, 28, 1]), 50)

# Run Tensorflow session
with tf.Session() as sess:

    # Increment the number of runs for saving to Tensorboard
    run_num = len(os.listdir(tensorboard_path)) + 1
    # Create Tensorboard writers for the training and test data
    train_writer = tf.summary.FileWriter('%s/%s/%s' % (tensorboard_path, 'run_' + str(run_num), 'train'), sess.graph)
    test_writer = tf.summary.FileWriter('%s/%s/%s' % (tensorboard_path, 'run_' + str(run_num), 'test'), sess.graph)

    summary = tf.summary.merge_all()

    # Initialise all the variables
    sess.run(tf.global_variables_initializer())

    # Train the model
    print("------------------------------------")
    print("Training model...")
    start_time = time.time()
    print("Training started: " + datetime.datetime.now().strftime("%b %d %T") + " for", num_epochs, "epochs")

    # Loop for number of training epochs
    for epoch in range(1, num_epochs + 1):

        # Initialise the iterator with the training data
        sess.run(iterator.initializer, feed_dict={images_placeholder: train_images, labels_placeholder: train_labels})

        # Loop over each training batch once per epoch
        train_loss = batch_loss = 0
        train_accuracy = batch_accuracy = 0
        train_summary = None
        while True:
            try:
                train_summary,  _, batch_loss, batch_accuracy = sess.run([summary, optimizer, loss, accuracy])
            except tf.errors.OutOfRangeError:
                break

        # Record training summaries
        train_loss += batch_loss
        train_accuracy += batch_accuracy
        train_writer.add_summary(train_summary, epoch)

        # Initialise the iterator with the test data
        sess.run(iterator.initializer, feed_dict={images_placeholder: test_images, labels_placeholder: test_labels})

        # Loop over each test batch once per epoch
        test_loss = batch_loss = 0
        test_accuracy = batch_accuracy = 0
        test_summary = None
        while True:
            try:
                test_summary, batch_loss, batch_accuracy = sess.run([summary, loss, accuracy])
            except tf.errors.OutOfRangeError:
                break

        # Record training summaries
        test_loss += batch_loss
        test_accuracy += batch_accuracy
        test_writer.add_summary(test_summary, epoch)

        # Display epoch statistics
        print("Epoch: {}/{} - "
              "Training loss: {:.3f}, acc: {:.2f} - "
              "Test loss: {:.3f}, acc: {:.2f}%".format(epoch, num_epochs, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100))

    end_time = time.time()
    print("Training took " + str(('%.3f' % (end_time - start_time))) + " seconds for", num_epochs, "epochs")

    # Evaluate the model
    # print("------------------------------------")
    # print("Evaluating model...")
    # Run on whole test set once
    # print("Accuracy: {:.2f}%".format(accuracy.eval(feed_dict={images: test_images, labels: test_labels}) * 100))
    # print("Number Correct: {}/{}".format((num_correct.eval(feed_dict={images: test_images, labels: test_labels})),
    #                                      num_test_examples))
