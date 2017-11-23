from sklearn.decomposition import PCA
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
    return score, y_pred_test


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
    return score
