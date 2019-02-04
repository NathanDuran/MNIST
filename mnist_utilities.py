from zipfile import ZipFile
from matplotlib import gridspec
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Load mnist data from zip
# Note: set is either 'test' or 'train'
def process_mnist(data_path, set):

    # Unzip and read csv files
    with ZipFile(data_path, 'r') as zipf:
        mnist_data = pd.read_csv(zipf.open('mnist_' + set + '.csv'), header=None)

        # First column are the labels
        labels = pd.get_dummies(mnist_data.iloc[:, 0]).values
        # Remaining columns are the image data
        images = mnist_data.iloc[:, 1:mnist_data.shape[1]].values

    return images, labels


# Display single digit from mnist data
def display_digit(images, labels, num):

    # Convert labels from one-hot
    label = np.where(labels[num] == 1)[0][0]
    # Reshape back to image dimensions
    image = images[num].reshape([28, 28])

    # Plot image
    plt.title('Example: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('bone'))
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
# Note: minimal spacing between each nodes plot and no titles, so works for for larger number of nodes (25+)
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
