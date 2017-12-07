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


def nc_fit(Xtrain, Xtest, Xtrain_lbls, Xtest_lbls, name, data, t0=time()):
    clf = NearestCentroid()
    # Train with the data
    clf.fit(Xtrain, Xtrain_lbls)

    # Create prediction for train data
    y_pred_test = clf.predict(Xtest)

    # How well does it fit
    score = clf.score(Xtest, Xtest_lbls)

    print('%-9s\t\t%.2fs\t%-9s\t%-9s'
          % (name, (time() - t0), score, data))

    # return score, y_pred_test


def pca_fit(Xtrain, Xtest, components):
    pca = PCA(n_components=components)
    pca_train = pca.fit_transform(Xtrain)
    pca_test = pca.fit(Xtrain).transform(Xtest)

    return pca_train, pca_test


def nn_fit(Xtrain, Xtest, Xtrain_lbls, Xtest_lbls, name, data, t0=time()):
    clf = neighbors.KNeighborsClassifier(1)
    clf.fit(Xtrain, Xtrain_lbls)
    Z = clf.predict(Xtest)
    score = clf.score(Xtest, Xtest_lbls)

    print('%-9s\t\t%.2fs\t%-9s\t%-9s'
          % (name, (time() - t0), score, data))
    # return score, Z


def nsc_fit(Xtrain, Xtrain_lbls, Xtest, Xtest_lbls, n_clust, rng, name, datat, t0=time()):
    centers = []
    labels = []
    correct_labels = []

    for i in range(0, rng):
        data = Xtrain[np.nonzero(Xtrain_lbls == i)]
        kmeans = KMeans(n_clusters=n_clust, random_state=42).fit(data)

        for k in range(0, n_clust):
            centers.append(kmeans.cluster_centers_[k, :])
            labels.append(str(i) + '_' + str(k))

    clf = NearestCentroid()
    clf.fit(centers, labels)
    pred = clf.predict(Xtest)

    for i in range(0, len(pred)):
        correct_labels.append(int(pred[i].split('_')[0]))

    score = accuracy_score(Xtest_lbls, correct_labels)

    print('%-9s\t\t%.2fs\t%-9s\t%-9s'
          % (name, (time() - t0), score, datat))
    # return score


def perceptron_sgd_plot(X, Y, LookFor):
    '''
    train perceptron and plot the total loss in each epoch.

    :param X: data samples
    :param Y: data labels
    :return: weight vector as a numpy array
    '''
    X = add_dummy_feature(X)

    w = np.zeros(len(X[0]))
    eta = 0.1
    n = 100
    errors = []
    bdata = []

    for t in range(n):
        total_error = 0
        tot = 0
        for i, x in enumerate(X):
            if(Y[i] == LookFor):
                y = 1
            else:
                y = -1

            if (np.dot(X[i], w) * Y[i]) <= 0:
                #w = w + eta * X[i] * Y[i]
                total_error += X[i] * y

        w += total_error*eta

        errors.append(tot)

   # print(bdata)
   # plt.plot(errors)
   # plt.xlabel('Iterations')
   # plt.ylabel('Total Errors')
   # plt.show()

    return w

# Make a prediction with weights

def predict(w, xtest, xtest_labels):
    k = np.asarray(w)
    

    test = add_dummy_feature(xtest).transpose()
    decision = np.dot(w,test)
    classification = np.argmax(decision,axis=0)

    try:
        score = accuracy_score(xtest_labels,classification)
    except ValueError:
        score = 0
    return score


def predict2(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0

# Estimate Perceptron weights using stochastic gradient descent


def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    errors = []
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict2(row, weights)
            error = row[-1] - prediction
            sum_error += error**2
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        errors.append(sum_error)


    plt.plot(errors)
    plt.xlabel('Iterations')
    plt.ylabel('Total Loss')
    plt.show()
    return weights


def binary_classifyer (XLables,wanted_label):
    newLabes = []

    for i, e in enumerate(XLables):
        if(int(e) == int(wanted_label)):
            newLabes.append(1)
        else:
            newLabes.append(-1)
    
    return newLabes
