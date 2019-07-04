import os
import time
import datetime
import tensorflow as tf
from mnist_utilities import display_weights_matrix_small, process_mnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow debugging

# Data and Tensorboard path
data_path = 'data/mnist.zip'
tensorboard_path = 'output/single_layer/tb_logs'
image_path = 'output/single_layer/images'

# Load training and test data
train_images, train_labels = process_mnist(data_path, 'train')
test_images, test_labels = process_mnist(data_path, 'test')

# Data parameters
input_size = train_images.shape[1]  # 784 (28 x 28)
num_classes = len(train_labels[0])  # 10
num_training_examples = train_images.shape[0]  # 60,000
num_test_examples = test_images.shape[0]  # 10,000

# Training parameters
learning_rate = 0.01
num_epochs = 1000

print("------------------------------------")
print("Using parameters...")
print("Input size: ", input_size)
print("Number of classes: ", num_classes)
print("Number of training examples:", num_training_examples)
print("Number of test examples:", num_test_examples)
print("Learning Rate: ", learning_rate)
print("Epochs: ", num_epochs)

# Define Tensorflow Graph
print("------------------------------------")
print('Define Graph...')

# Image and label placeholder variables
images_placeholder = tf.placeholder(tf.float32, [None, input_size], name='images')
labels_placeholder = tf.placeholder(tf.float32, [None, num_classes], name='labels')

# Define single layer network
with tf.name_scope('nn_layer'):
    weights = tf.Variable(tf.zeros([input_size, num_classes], dtype='float32'), name='weights')
    biases = tf.Variable(tf.zeros([num_classes], dtype='float32'), name='biases')
    Wx_plus_b = tf.matmul(images_placeholder, weights) + biases
    output_prediction = tf.nn.sigmoid(Wx_plus_b, name='activation')

# Calculate the cost
with tf.name_scope('cross_entropy'):
    loss = tf.losses.softmax_cross_entropy(labels_placeholder, logits=output_prediction)
    loss_summary = tf.summary.scalar('Loss', loss)

# Minimise the cost with optimisation function
with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Calculate the average of correct predictions
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(output_prediction, axis=1), tf.argmax(labels_placeholder, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_summary = tf.summary.scalar('Accuracy', accuracy)

# Create images for each nodes weights
with tf.name_scope('tb_images'):
    weight_images = tf.reshape(tf.transpose(weights), [num_classes, 28, 28, 1])
    image_summary = tf.summary.image('hidden_weights', weight_images, max_outputs=num_classes)

with tf.Session() as sess:

    # Remove old Tensorboard directory
    if tf.gfile.Exists(tensorboard_path):
        tf.gfile.DeleteRecursively(tensorboard_path)

    # Create Tensorboard writers for the training and test data
    train_writer = tf.summary.FileWriter('%s/%s' % (tensorboard_path, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter('%s/%s' % (tensorboard_path, 'test'), sess.graph)
    summary = tf.summary.merge([accuracy_summary, loss_summary])

    # Initialise all the variables
    sess.run(tf.global_variables_initializer())

    # Train the model
    print("------------------------------------")
    print("Training model...")
    start_time = time.time()
    print("Training started: " + datetime.datetime.now().strftime("%b %d %T") + " for", num_epochs, "epochs")

    # Loop for number of training epochs
    for epoch in range(1, num_epochs + 1):

        # Loop over each training batch once per epoch
        _, train_loss, train_accuracy, train_summary = sess.run([optimizer, loss, accuracy, summary], feed_dict={images_placeholder: train_images, labels_placeholder: train_labels})

        # Loop over each test batch once per epoch
        test_loss, test_accuracy, test_summary = sess.run([loss, accuracy, summary], feed_dict={images_placeholder: test_images, labels_placeholder: test_labels})

        # Record training and image summaries
        train_writer.add_summary(train_summary, epoch)
        test_writer.add_summary(test_summary, epoch)
        test_writer.add_summary(image_summary.eval(), epoch)

        # Display learned weights
        if epoch < 100 and epoch % 10 == 0 or epoch >= 100 and epoch % 100 == 0:
            display_weights_matrix_small(weights, num_classes, epoch, test_accuracy, save=True, path=image_path)

        # Display epoch statistics
        print("Epoch: {}/{} - "
              "Training loss: {:.3f}, acc: {:.2f} - "
              "Test loss: {:.3f}, acc: {:.2f}%".format(epoch, num_epochs, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100))

    end_time = time.time()
    print("Training took " + str(('%.3f' % (end_time - start_time))) + " seconds for", num_epochs, "epochs")


