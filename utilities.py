import matplotlib.pyplot as plt
from matplotlib import gridspec
import tensorflow as tf


# Creates dataset and iterator placeholders for feeding data into the network
def make_dataset_placeholder(input_size, num_classes, batch_size, name='dataset'):

    with tf.name_scope(name):
        # Image and label placeholder variables
        images_placeholder = tf.placeholder(tf.float32, [None, input_size], name='images_placeholder')
        labels_placeholder = tf.placeholder(tf.float32, [None, num_classes], name='labels_placeholder')

        # Make Tensorflow dataset placeholder
        dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder)).batch(batch_size).repeat(1)

        # Make iterator placeholder
        iterator = dataset.make_initializable_iterator()

        return iterator, images_placeholder, labels_placeholder


# Attach a lot of summaries to a Tensor for TensorBoard visualization
def variable_summaries(var):

    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# Re-usable neural network layer
# Note: If summary set to true, lots of summary data is recorded and may default to CPU
def nn_layer(input_tensor, input_dim, output_dim, activation_func=tf.nn.relu, layer_name='', summary=False):

    # Name scope for entire layer
    with tf.name_scope(layer_name):

        # Weights
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal(shape=[input_dim, output_dim], stddev=0.1))
            if summary:
                variable_summaries(weights)

        # Bias
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.1, shape=[output_dim]))
            if summary:
                variable_summaries(biases)

        # (Weight * Input) + Bias
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            if summary:
                tf.summary.histogram('pre_activations', preactivate)

        # Activation function
        with tf.name_scope('activation'):
            activation = activation_func(preactivate, name='activation')
            if summary:
                tf.summary.histogram('activation', activation)

        return activation, weights


# Display single digit from mnist data
def display_digit(images, labels, num):

    label = labels[num]
    image = images[num].reshape([28, 28])
    plt.title('Example: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()


# Display weights for each node in the input weight_matrix
# Note: includes title corresponding to node number so works best for small number of nodes (10 - 25)
def display_weights(weights_matrix, num_hidden_nodes, epoch, accuracy, save=False, path=''):

    # Calculate number of rows to display
    num_cols = 5
    num_rows = num_hidden_nodes / num_cols

    # Create figure
    fig = plt.figure(figsize=(num_cols, num_rows + 1))

    # Loop over all hidden nodes and add to the chart
    for i in range(num_hidden_nodes):

        # Each node is a column of the weight_matrix
        node_weights = weights_matrix[:, i].eval()

        # Create plot and add image
        plt.subplot(num_rows, num_cols, i + 1)
        plt.title(i)
        plt.imshow(node_weights.reshape([28, 28]), cmap=plt.get_cmap('seismic'))
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)

    # Add title to figure
    fig.suptitle("Epoch: {}     Accuracy: {:.2f}".format(epoch, accuracy * 100))
    # Adjust subplots to make room for title
    fig.subplots_adjust(bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)

    # Save and display
    if save:
        fig.savefig(path + '/Epoch - ' + str(epoch) + '.png')
    fig.show()


# Display weights for each node in the input weight_matrix
# Note: minimal spacing between each nodes plot and no titles, so works for  for larger number of nodes (25+)
def display_weights_matrix(weights_matrix, num_hidden_nodes, epoch, accuracy, save=False, path=''):

    # Calculate number of rows to display
    num_cols = 5
    num_rows = int(num_hidden_nodes / num_cols)

    # Create figure
    fig = plt.figure(figsize=(5, 5))

    # Define grid spacing
    gs = gridspec.GridSpec(num_rows, num_cols,
                           wspace=0.1, hspace=0.1,
                           top=1. - 0.5 / (num_rows + 1), bottom=0.5 / (num_rows + 1),
                           left=0.5 / (num_cols + 1), right=1 - 0.5 / (num_cols + 1))

    # Loop over all hidden nodes and add to the chart
    node = 0
    for i in range(0, num_rows):
        for j in range(0, num_cols):

            # Each node is a column of the weight_matrix
            node_weights = weights_matrix[:, node].eval()

            # Create plot and add image
            ax = plt.subplot(gs[i, j])
            ax.imshow(node_weights.reshape([28, 28]), cmap=plt.get_cmap('seismic'))
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
            node += 1

    # Add title to figure
    fig.suptitle("Epoch: {}     Accuracy: {:.2f}".format(epoch, accuracy * 100))
    # Adjust subplots to make room for title
    fig.subplots_adjust(bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)

    # Save and display
    if save:
        fig.savefig(path + '/Epoch - ' + str(epoch) + '.png')
    fig.show()
