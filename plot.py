import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib
from scipy import ndimage


def plot_data(data, labels, title="", index=0):
    labels = np.ndarray.flatten(labels)
    classes = list(set(labels))
    N = len(classes)


#    cmap =  cm.get_cmap('rainbow')  #c
 #   cmaplist = [cmap(i) for i in range(N)]
 #   cmap = cmap.from_list('Custom cmap', cmaplist, N)

    cmap2 = cm.rainbow(np.linspace(0, 1, N))

    plots = []
    plt.figure(index)
    plt.title(title)
    for i, label in enumerate(classes):
        class_data = np.asarray(
            [x for j, x in enumerate(data) if labels[j] == label])
        x = class_data[:, 0]
        y = class_data[:, 1]
        plots.append(plt.scatter(x, y, color=cmap2[i]))

    plt.legend(plots,
               classes, scatterpoints=1, loc='upper right', ncol=2, fontsize=8)
    plt.draw()
    plt.show()


def plot_mnist_digit(images):
    first_image = images[4]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

def plot_orl_face(images):
    first_image = images[8]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((30, 40))
    plt.imshow(ndimage.rotate(pixels,270) , cmap='gray')
    plt.show()



def plot_mnist_n_by_m(elts, m, n):
    """Plot MNIST images in an m by n table. Note that we crop the images
    so that they appear reasonably close together.  Note that we are
    passed raw MNIST data and it is reshaped.
    """
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

def plot_orl_n_by_m(elts, m, n):
    """Plot ORL images in an m by n table. Note that we crop the images
    so that they appear reasonably close together.  Note that we are
    passed raw ORL data and it is reshaped.
    """

    fig = plt.figure()
    images = [elt.reshape(30, 40) for elt in elts]
    img = np.concatenate([np.concatenate([images[m * y + x] for x in range(m)], axis=1)
                          for y in range(n)], axis=0)
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(ndimage.rotate(img,270), cmap=matplotlib.cm.binary)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()
