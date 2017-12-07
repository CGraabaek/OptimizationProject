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
from time import time


def nc_fit(Xtrain,Xtest,Xtrain_lbls,Xtest_lbls,name,data):
    t0 = time()
    clf = NearestCentroid()
    # Train with the data
    clf.fit(Xtrain,Xtrain_lbls)

    # Create predition for train data
    y_pred_test = clf.predict(Xtest)

    #How well does it fit
    score = clf.score(Xtest,Xtest_lbls)


  

    print('%-9s\t%.2fs\t%i\t%-9s'
          % (name, (time()-t0),score,data))

    ##return score, y_pred_test

def pca_fit(Xtrain,Xtest,components):
    pca = PCA(n_components=components)
    PCA_train = pca.fit_transform(Xtrain)
   # PCA_test = pca.fit(Xtrain).transform(Xtest)
    PCA_test = pca.transform(Xtest)

    return PCA_train, PCA_test

def nn_fit(Xtrain,Xtest,Xtrain_lbls,Xtest_lbls):
    clf = neighbors.KNeighborsClassifier(1)
    clf.fit(Xtrain, Xtrain_lbls)
    Z = clf.predict(Xtest)
    score = clf.score(Xtest,Xtest_lbls) 
    return score, Z


def nsc_fit(Xtrain,Xtrain_lbls,Xtest,Xtest_lbls,n_clust,rng):
    centers = []
    labels = []
    new_labels = []
    correct_labels = []

    for i in range(0,rng):
        data = Xtrain[np.nonzero(Xtrain_lbls==i)]
        kmeans = KMeans(n_clusters=n_clust,random_state=42).fit(data)

        for k in range(0,n_clust):
            centers.append(kmeans.cluster_centers_[k,:])
            labels.append(str(i) +  '_' + str(k))
 
    clf = NearestCentroid()


    clf.fit(centers,labels)
    predict = clf.predict(Xtest)

    for i in range(0,len(predict)):
       correct_labels.append(int(predict[i].split('_')[0]))

    score = accuracy_score(Xtest_lbls,correct_labels)
    
    return score


def cluster(X) :
    kmeans = KMeans(n_clusters = 3, random_state = 0)                   
    kmeans.fit(X)                  
    plt.scatter(X[:, 0], X[:, 1],c = kmeans.labels_)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 50, c = 'yellow') 
    plt.show()


