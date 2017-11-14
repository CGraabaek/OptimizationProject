import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from mnist import MNIST
from sklearn.neighbors import NearestCentroid
from sklearn import neighbors


def nc_fit(Xtrain,Xtest,Xtrain_lbls,Xtest_lbls):
    clf = NearestCentroid()
    # Train with the data
    clf.fit(Xtrain,Xtrain_lbls)

    # Create predition for train data
    y_pred_test = clf.predict(Xtest)

    #How well does it fit
    score = clf.score(Xtest,Xtest_lbls)
    return score


def pca_fit(Xtrain,Xtrain_lbls,Xtest,Xtest_lbls, components):
    pca = PCA(n_components=components)
    PCA_train = pca.fit_transform(Xtrain)
    PCA_test = pca.fit(Xtrain).transform(Xtest)
    return PCA_train, PCA_test

def nn_fit(Xtrain,Xtest,Xtrain_lbls,Xtest_lbls):
    clf = neighbors.KNeighborsClassifier()
    clf.fit(Xtrain, Xtrain_lbls)
    Z = clf.predict(Xtest)
    score = clf.score(Xtest,Xtest_lbls)
    return score
