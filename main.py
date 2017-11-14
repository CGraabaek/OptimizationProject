import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mnist import MNIST
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split
import classifiers

print('--------------------- Nearest Centroid - ORL ---------------------')
orl_data = np.array(sio.loadmat('orl_data.mat')['data']).transpose()
orl_labels = np.array(sio.loadmat('orl_lbls.mat')['lbls']).ravel()

orl_data_train, orl_data_test, orl_lables_train, orl_labels_test = train_test_split(orl_data,orl_labels,test_size=0.30,random_state=42)

orl_score = classifiers.nc_fit(orl_data_train,orl_data_test,orl_lables_train,orl_labels_test)

###-----PCA------###

orl_pca_train, orl_pca_test = classifiers.pca_fit(orl_data_train,orl_lables_train,orl_data_test,orl_labels_test,2)
orl_pca_train_10, orl_pca_test_10 = classifiers.pca_fit(orl_data_train,orl_lables_train,orl_data_test,orl_labels_test,10)
orl_pca_score_2 = classifiers.nc_fit(orl_pca_train,orl_pca_test,orl_lables_train,orl_labels_test)
orl_pca_score_10 = classifiers.nc_fit(orl_pca_train_10,orl_pca_test_10,orl_lables_train,orl_labels_test)


print('Score from function ' + str(orl_score))
print('Score from PCA - 2 Components ' + str(orl_pca_score_2))
print('Score from PCA - 10 Components ' + str(orl_pca_score_10))



print('--------------------- Nearest Centroid - MINST ---------------------')
mndata = MNIST('./', return_type='numpy')
images_training, labels_training = mndata.load_training()
images_test, labels_test = mndata.load_testing()

minst_score = classifiers.nc_fit(images_training,images_test,labels_training,labels_test)

minst_pca_train, minst_pca_test = classifiers.pca_fit(images_training,labels_training,images_test,labels_test,2)
minst_pca_train_10, minst_pca_test_10 = classifiers.pca_fit(images_training,labels_training,images_test,labels_test,10)
minst_pca_score_2 = classifiers.nc_fit(minst_pca_train,minst_pca_test,labels_training,labels_test)
minst_pca_score_10 = classifiers.nc_fit(minst_pca_train_10,minst_pca_test_10,labels_training,labels_test)


print('Score from function ' + str(minst_score))
print('Score from PCA - 2 Components ' + str(minst_pca_score_2))
print('Score from PCA - 10 Components ' + str(minst_pca_score_10))


print('--------------------- Nearest Neighbor - ORL ---------------------')

orl_score_nc = classifiers.nn_fit(orl_data_train,orl_data_test,orl_lables_train,orl_labels_test)
print('Score from function ' + str(orl_score_nc))
