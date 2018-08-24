import matplotlib.pyplot as plt


def display_digit(x, y, num):

    label = y[num].argmax(axis=0)
    image = x[num].reshape([28, 28])
    plt.title('Example: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()