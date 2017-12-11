import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib
from scipy import ndimage
from sklearn import metrics

# For plotting a scatter plot of the data


def plot_data(data, labels, title="", index=0):
    labels = np.ndarray.flatten(labels)
    classes = list(set(labels))
    N = len(classes)

    # Set color map
    cmap2 = cm.rainbow(np.linspace(0, 1, N))

    plots = []
    plt.figure(index)
    plt.title(title)
    for i, label in enumerate(classes):
        class_data = np.asarray(
            [x for j, x in enumerate(data) if labels[j] == label])
        x = class_data[:, 0]
        y = class_data[:, 1]
        plots.append(plt.scatter(x, y, color=cmap2[i], s=5))

    plt.legend(plots,
               classes, scatterpoints=1, loc='upper right', ncol=2, fontsize=8)
    plt.draw()
    plt.show()

# Plot a single mnist digit


def plot_mnist_digit(images, index):
    first_image = images[index]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

# Plot at single ORL face


def plot_orl_face(images, indes):
    first_image = images[index]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((30, 40))
    plt.imshow(ndimage.rotate(pixels, 270), cmap='gray')
    plt.show()

# Create an m by n grid of digits


def plot_mnist_n_by_m(elts, m, n):
    fig = plt.figure()
    images = [elt.reshape(28, 28) for elt in elts]
    img = np.concatenate([np.concatenate([images[m * y + x] for x in range(m)], axis=1)
                          for y in range(n)], axis=0)
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(img, cmap=matplotlib.cm.binary)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()


# Create an m by n grid of faces
def plot_orl_n_by_m(elts, m, n):
    fig = plt.figure()
    images = [elt.reshape(30, 40) for elt in elts]
    img = np.concatenate([np.concatenate([images[m * y + x] for x in range(m)], axis=1)
                          for y in range(n)], axis=0)
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(ndimage.rotate(img, 270), cmap=matplotlib.cm.binary)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()


def plot_confusion_matrix(Xtest_lbls, predicted, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = metrics.confusion_matrix(Xtest_lbls, predicted)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# For plotting different kinds of same label


def plot_numbers(x, labels, wanted_label):
    elts = []

    for i in range(0, 1000):
        if(labels[i] == wanted_label):
            elts.append(x[i])

    plot_mnist_n_by_m(elts, 5, 2)



# Plot a digit from 0 - 9
def plot_images_separately(imgs):

    images = []
    images.append(imgs[3])
    images.append(imgs[2])
    images.append(imgs[1])
    images.append(imgs[30])
    images.append(imgs[4])
    images.append(imgs[8])
    images.append(imgs[11])
    images.append(imgs[0])
    images.append(imgs[61])
    images.append(imgs[7])

    images2 = [elt.reshape(28, 28) for elt in images]

    fig = plt.figure()
    for j in xrange(1, 11):
        ax = fig.add_subplot(1, 10, j)
        ax.matshow(images2[j-1], cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.show()
