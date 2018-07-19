import os
import gzip
import pickle
import utilities as utils
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow debugging
########
# https://www.oreilly.com/learning/not-another-mnist-tutorial-with-tensorflow

data_path = 'data/'

# Load training and test data
with gzip.open(data_path + 'mnist.pkl', 'rb') as file:
    dataset = pickle.load(file)

train_images = dataset['train_images']
train_labels = dataset['train_labels']
test_images = dataset['test_images']
test_labels = dataset['test_labels']

input_dim = 784  # 28 x 28
num_hidden_nodes = 500
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

# Display a random digit
# utils.display_digit(train_x, train_y, np.random.randint(0, train_x.shape[0]))

x = tf.placeholder(tf.float32, [None, input_dim])
y = tf.placeholder(tf.int64)


def neural_network_model(input_data):

    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([input_dim, num_hidden_nodes])),
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


def train_neural_network(input_data):

    # Pass data through the model
    prediction = neural_network_model(input_data)

    # Calculate the cost
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y) ) / train_x.shape[0]
    cost = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(y, prediction))

    # Minimise the cost with optimisation function
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.Session() as sess:
        # Initialise all the variables
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            epoch_loss = 0
            # for _ in range(int(mnist.train.num_examples / batch_size)):
            #     epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: train_images, y: train_labels})

            epoch_loss = c

            print("Epoch", epoch + 1, "completed out of", num_epochs, "loss:", epoch_loss)

        # Count how many predictions (index of highest probability) match the labels
        correct = tf.equal(tf.argmax(prediction, axis=1), y)

        # Count how many are correct
        num_correct = tf.reduce_sum(tf.cast(correct, tf.int32))

        # Calculate the average for correct predictions
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        print("Number Correct: " + str(num_correct.eval(feed_dict={x: test_images, y: test_labels})) + " / " + str(test_images.shape[0]))
        print("Accuracy: " + str(accuracy.eval(feed_dict={x: test_images, y: test_labels})))


train_neural_network(x)