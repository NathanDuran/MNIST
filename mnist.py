import os
import time
import datetime
import gzip
import pickle
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow debugging
########
# https://www.oreilly.com/learning/not-another-mnist-tutorial-with-tensorflow

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
num_hidden_nodes = 500

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

# Image and label placeholder variables
images_placeholder = tf.placeholder(tf.float32, [None, input_size])
labels_placeholder = tf.placeholder(tf.int32)

# Make Tensorflow dataset and iterator placeholder
dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder)).batch(batch_size).repeat(1)
iterator = dataset.make_initializable_iterator()
images, labels = iterator.get_next()

# Define placeholder summary variables for Tensorboard
with tf.name_scope('performance'):
    # Loss placeholder
    loss_placeholder = tf.placeholder(tf.float32, shape=None, name='loss_summary')
    # Create a scalar summary object for the training and test loss
    train_loss_summary = tf.summary.scalar('train_loss', loss_placeholder)
    test_loss_summary = tf.summary.scalar('test_loss', loss_placeholder)

    # Accuracy placeholder
    accuracy_placeholder = tf.placeholder(tf.float32, shape=None, name='accuracy_summary')
    # Create a scalar summary object for the training and test accuracy
    train_accuracy_summary = tf.summary.scalar('train_accuracy', accuracy_placeholder)
    test_accuracy_summary = tf.summary.scalar('test_accuracy', accuracy_placeholder)

    # Merge summaries together
    train_summaries = tf.summary.merge([train_loss_summary, train_accuracy_summary])
    test_summaries = tf.summary.merge([test_loss_summary, test_accuracy_summary])


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
    l1 = tf.nn.relu(l1, name='hidden_1')

    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2, name='hidden_2')

    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3, name='hidden_3')

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


# Define Tensorflow Graph
print("------------------------------------")
print('Define Graph...')
# Pass data through the model
prediction = neural_network_model(images)

# Calculate the cost
cost = tf.reduce_sum(tf.losses.sparse_softmax_cross_entropy(labels, prediction), name='loss')

# Minimise the cost with optimisation function
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Count how many predictions (index of highest probability) match the labels
correct = tf.equal(tf.argmax(prediction, axis=1), tf.cast(labels, tf.int64))

# Count how many are correct
num_correct = tf.reduce_sum(tf.cast(correct, tf.int32))

# Calculate the average of correct predictions
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

# Run Tensorflow session
with tf.Session() as sess:
    # Increment the number of runs for saving to Tensorboard
    run_num = len(os.listdir(tensorboard_path)) + 1
    # Create Tensorboard writers for the training and test data
    train_writer = tf.summary.FileWriter('%s/%s/%s' % (tensorboard_path, 'run_' + str(run_num), 'train'), sess.graph)
    test_writer = tf.summary.FileWriter('%s/%s/%s' % (tensorboard_path, 'run_' + str(run_num), 'test'), sess.graph)

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
        while True:
            try:
                _, batch_loss, batch_accuracy = sess.run([optimizer, cost, accuracy])
            except tf.errors.OutOfRangeError:
                break

        train_loss += batch_loss
        train_accuracy += batch_accuracy

        # Record training summaries
        train_summary = sess.run(train_summaries,
                                 feed_dict={loss_placeholder: train_loss, accuracy_placeholder: train_accuracy})
        train_writer.add_summary(train_summary, epoch)

        # Initialise the iterator with the test data
        sess.run(iterator.initializer, feed_dict={images_placeholder: test_images, labels_placeholder: test_labels})

        # Loop over each test batch once per epoch
        test_loss = batch_loss = 0
        test_accuracy = batch_accuracy = 0
        while True:
            try:
                batch_loss, batch_accuracy, batch_num_correct = sess.run([cost, accuracy, num_correct])
            except tf.errors.OutOfRangeError:
                break
        test_loss += batch_loss
        test_accuracy += batch_accuracy

        # Record training summaries
        test_summary = sess.run(test_summaries,
                                feed_dict={loss_placeholder: test_loss, accuracy_placeholder: test_accuracy})
        test_writer.add_summary(test_summary, epoch)

        # Display epoch statistics
        print("Epoch: {}/{} - "
              "Training loss: {:.3f}, acc: {:.2f} - "
              "Test loss: {:.3f}, acc: {:.2f}%".format(epoch,
                                                       num_epochs,
                                                       train_loss,
                                                       train_accuracy * 100,
                                                       test_loss,
                                                       test_accuracy * 100))

    end_time = time.time()
    print("Training took " + str(('%.3f' % (end_time - start_time))) + " seconds for", num_epochs, "epochs")

    # Evaluate the model
    print("------------------------------------")
    print("Evaluating model...")
    # Run on whole test set once
    print("Accuracy: {:.2f}%".format(accuracy.eval(feed_dict={images: test_images, labels: test_labels}) * 100))
    print("Number Correct: {}/{}".format((num_correct.eval(feed_dict={images: test_images, labels: test_labels})),
                                         num_test_examples))
