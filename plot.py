import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap


def plot_data(data,labels,title=""):
    labels = np.ndarray.flatten(labels)
    classes = list(set(labels))
    N = len(classes)

    cmap =  cm.get_cmap('rainbow')  #c
    cmaplist = [cmap(i) for i in range(N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, N)

    cmap2= cm.rainbow(np.linspace(0, 1, N))

    plots = []
    plt.figure()
    plt.title(title)
    for i, label in enumerate(classes):
        class_data =  np.asarray([x for j, x in enumerate(data) if labels[j]==label])
        x = class_data[:,0]
        y = class_data[:,1]
        plots.append(plt.scatter(x, y ,color=cmap2[i]))

    plt.legend(plots,
               classes
               ,scatterpoints=1
               ,loc='upper right'
               ,ncol=2
               ,fontsize=8)
    plt.draw()
    plt.show()



def plot_mnist(images): 
    first_image = images[4]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

def plot_test(data,labels,title=""):
    


    X_r = data
    target_names = labels
    
    plt.figure()
    for c, i, target_name in zip("rgb", [0, 1, 2, 3, 4, 5, 6, 7, 8], target_names):
        plt.scatter(X_r[y==i,0], X_r[y==i,1], c=c, label=target_name)
    plt.legend()
    plt.title('PCA of IRIS dataset')

    plt.show()