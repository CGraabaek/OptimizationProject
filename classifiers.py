from sklearn.decomposition import PCA
from sklearn.neighbors import NearestCentroid
from sklearn import neighbors
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.stats import mode
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import add_dummy_feature
from time import time
import plot


def nc_fit(Xtrain, Xtest, Xtrain_lbls, Xtest_lbls, name, data, t0=time()):
    #Create a nearest centroid
    clf = NearestCentroid()
    # Train with the data
    clf.fit(Xtrain, Xtrain_lbls)

    # Create prediction for test data
    y_pred_test = clf.predict(Xtest)

    # How well does it fit
    score = clf.score(Xtest, Xtest_lbls)

    print('%-9s\t%.2fs\t%-9s\t%-9s'
          % (name, (time() - t0), score, data))

    return y_pred_test


def pca_fit(Xtrain, Xtest, components):
    #Set pca components
    pca = PCA(n_components=components)
    #Create the PCA training set
    pca_train = pca.fit_transform(Xtrain)
    #Create tje PCA test set
    pca_test = pca.fit(Xtrain).transform(Xtest)

    return pca_train, pca_test


def nn_fit(Xtrain, Xtest, Xtrain_lbls, Xtest_lbls, name, data, t0=time()):
    ## Create the nearest neighbor classifier
    clf = neighbors.KNeighborsClassifier(1)
    #Fit the data
    clf.fit(Xtrain, Xtrain_lbls)
    #Predict the labels
    Z = clf.predict(Xtest)
    #Calculate the score
    score = clf.score(Xtest, Xtest_lbls)

    print('%-9s\t%.2fs\t%-9s\t%-9s'
          % (name, (time() - t0), score, data))
    # return score, Z


def nsc_fit(Xtrain, Xtrain_lbls, Xtest, Xtest_lbls, n_clust, rng, start, name, datat, t0=time()):
    centers = []
    labels = []
    correct_labels = []

    #Cluster the data
    # Start is the index, it starts at since MNIST starts at 0 and ORL at 1.
    # rng is the range, MNIST has 10 classes where ORL has 40.
    for i in range(start, rng):
        data = Xtrain[np.nonzero(Xtrain_lbls == i)]
        kmeans = KMeans(n_clusters=n_clust, random_state=42).fit(data)

        # Get the centers for the amount of clusters specified.
        for k in range(0, n_clust):
            centers.append(kmeans.cluster_centers_[k, :])
            labels.append(str(i) + '_' + str(k))

    #Fit with nearest centroid
    clf = NearestCentroid()
    clf.fit(centers, labels)
    pred = clf.predict(Xtest)

    for i in range(0, len(pred)):
        correct_labels.append(int(pred[i].split('_')[0]))

    #Calculate score
    score = accuracy_score(Xtest_lbls, correct_labels)

    print('%-9s\t%.2fs\t%-9s\t%-9s'
          % (name, (time() - t0), score, datat))
    return pred

def batch_perceptron_train_and_classify(data,lables,rng,name,datat,t0=time()):
    # Create array for the weights for each class.
    weight_array = [] 

    for i in range(0, rng):
        weight_array.append(batch_perceptron(data, lables,i,0.1,30,False))

    score = mse_classify(weight_array, data, lables)
    print('%-9s\t%.2fs\t%-9s\t%-9s'
          % (name, (time() - t0), score, datat))

def mse_perceptron_train_and_classify(data,lables,rng,name,datat,t0=time()):
    weight_array =  []#np.empty((10,784),dtype=float) #[]
    XtrainPinv = np.linalg.pinv(data)

    for i in range(0,rng):
        new_lbls = binary_classifyer(lables,i)
        weight_array.append(mse_train(XtrainPinv,new_lbls))

    score = mse_classify(weight_array,data,lables)
    print('%-9s\t%.2fs\t%-9s\t%-9s'
          % (name, (time() - t0), score, datat))
    



def batch_perceptron(xx, xlabels, wanted_label, learning_rate=0.05, max_t=200, debug=False):
  x = []
  labels = []

  for i, img in enumerate(xx):
    #img = np.append(img, [1])
    x.append(img)

    label = -1
    if int(xlabels[i]) is int(wanted_label):
      label = 1
    labels.append( label )

  N = len(x[0]) #Get length of data
  w = np.zeros(N) # initialize weights to all zeros

  # loop until max_t - 
  for t in range(0, max_t):
    delta_sum = np.zeros(N)
    errors = 0

    for i in range(0, len(labels)):
      dot = np.dot(x[i], w) * labels[i]

      if dot <= 0:
        delta_sum += labels[i] * x[i]
        errors += 1

    w += learning_rate * delta_sum

    if debug:
      print("epoch=" + str(t) + " error=" + str(errors))
  return w

def binary_classifyer(XLables, wanted_label):
    newLabels = []

    # Loop through labels and add a 1 to the wanted labels, -1 to the others
    for i, e in enumerate(XLables):
        if(int(e) == int(wanted_label)):
            newLabels.append(1)
        else:
            newLabels.append(-1)

    return newLabels


def mse_classify(w, xtest, xtest_labels):
     classified_labels = []

     for k in range(len(xtest)):
        lbl = None
        old = None

        for i in range(len(w)):
            decision = np.dot(w[i], xtest[k])
            # print(decision)
            if(old == None or decision > old):
                old = decision
                lbl = i

        classified_labels.append(lbl)
     score = accuracy_score(xtest_labels, classified_labels)
     return score

def mse_train(pinv,lbls):
    #As input the pseudo inverse of the data is taken.
    return np.dot(pinv, lbls)
