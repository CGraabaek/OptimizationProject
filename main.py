import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from mnist import MNIST
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split
import classifiers
import plot
from time import time
from sklearn import metrics



## Setup data
#ORL
orl_data = np.array(sio.loadmat('orl_data.mat')['data']).transpose()
orl_labels = np.array(sio.loadmat('orl_lbls.mat')['lbls']).ravel()
orl_data_train, orl_data_test, orl_lables_train, orl_labels_test = train_test_split(orl_data,orl_labels,test_size=0.30,random_state=42,stratify=orl_labels)
orl_pca_train, orl_pca_test = classifiers.pca_fit(orl_data_train,orl_data_test,2)
orl_pca_train_10, orl_pca_test_10 = classifiers.pca_fit(orl_data_train,orl_data_test,10)
#MINST
mndata = MNIST('./', return_type='numpy')
images_training, labels_training = mndata.load_training()
images_test, labels_test = mndata.load_testing()

minst_pca_train, minst_pca_test = classifiers.pca_fit(images_training,images_test,2)
minst_pca_train_10, minst_pca_test_10 = classifiers.pca_fit(images_training,images_test,10)


print(82 * '_')
print("Classifiers")
print(82 * '_')
print('name\t\t\t\ttime\tscore\t\tdata')
print(82 * '_')

classifiers.nc_fit(orl_data_train,orl_data_test,orl_lables_train,orl_labels_test, "Nearest Centroid", "ORL")
classifiers.nc_fit(images_training,images_test,labels_training,labels_test,"Nearest Centroid","MNIST")
classifiers.nn_fit(orl_data_train,orl_data_test,orl_lables_train,orl_labels_test,"Nearest Neighbor","ORL")
#classifiers.nn_fit(images_training,images_test,labels_training,labels_test, "NN","MNIST")
classifiers.nsc_fit(images_training,labels_training,images_test,labels_test,2,10,0,"NSC - 2 Clusters","MNIST")
classifiers.nsc_fit(orl_data_train,orl_lables_train,orl_data_test,orl_labels_test,2,40,1,"NSC - 2 Clusters","ORL")
classifiers.nsc_fit(images_training,labels_training,images_test,labels_test,3,10,0,"NSC - 3 Clusters","MNIST")
classifiers.nsc_fit(orl_data_train,orl_lables_train,orl_data_test,orl_labels_test,3,40,1,"NSC - 3 Clusters","ORL")
classifiers.nsc_fit(images_training,labels_training,images_test,labels_test,5,10,0,"NSC - 5 Clusters","MNIST")
classifiers.nsc_fit(orl_data_train,orl_lables_train,orl_data_test,orl_labels_test,5,40,1,"NSC - 5 Clusters","ORL")
classifiers.batch_perceptron_train_and_classify(orl_data_train,orl_lables_train,40,"Batch Perceptron - ORL","ORL")
classifiers.batch_perceptron_train_and_classify(images_training,labels_training,10,"Batch Perceptron - MNIST","MNIST")
classifiers.mse_perceptron_train_and_classify(orl_data_train,orl_lables_train,40,"MSE Perceptron","ORL")
classifiers.mse_perceptron_train_and_classify(images_training,labels_training,10,"MSE Perceptron" ,"MNIST")



print(82 * '_')
print("PCA - 2 Components")
print(82 * '_')
classifiers.nc_fit(orl_pca_train,orl_pca_test,orl_lables_train,orl_labels_test, "Nearest Centroid - PCA 2","ORL")
classifiers.nc_fit(minst_pca_train,minst_pca_test,labels_training,labels_test,"Nearest Centroid - PCA 2","MNIST")
classifiers.nn_fit(orl_pca_train,orl_pca_test,orl_lables_train,orl_labels_test,"Nearest Neighbor - PCA 2","ORL")
classifiers.nn_fit(minst_pca_train,minst_pca_test,labels_training,labels_test, "NN - PCA 2 ","MNIST")
classifiers.nsc_fit(minst_pca_train,labels_training,minst_pca_test,labels_test,2,10,0,"NSC - 2 Clusters - PCA 2","MNIST")
classifiers.nsc_fit(minst_pca_train,labels_training,minst_pca_test,labels_test,3,10,0,"NSC - 3 Clusters - PCA 2","MNIST")
classifiers.nsc_fit(minst_pca_train,labels_training,minst_pca_test,labels_test,5,10,0,"NSC - 5 Clusters - PCA 2","MNIST")
classifiers.nsc_fit(orl_pca_train,orl_lables_train,orl_pca_test,orl_labels_test,2,40,1,"NSC - 2 Clusters - PCA 2","ORL")
classifiers.nsc_fit(orl_pca_train,orl_lables_train,orl_pca_test,orl_labels_test,3,40,1,"NSC - 3 Clusters - PCA 2","ORL")
classifiers.nsc_fit(orl_pca_train,orl_lables_train,orl_pca_test,orl_labels_test,5,40,1,"NSC - 5 Clusters - PCA 2","ORL")
classifiers.batch_perceptron_train_and_classify(orl_pca_train,orl_lables_train,40,"Batch Perceptron - PCA 2","ORL")
classifiers.batch_perceptron_train_and_classify(minst_pca_train,labels_training,10,"Batch Perceptron - PCA 2","MNIST")
classifiers.mse_perceptron_train_and_classify(orl_pca_train,orl_lables_train,40,"MSE Perceptron - PCA 2","ORL")
classifiers.mse_perceptron_train_and_classify(minst_pca_train,labels_training,10,"MSE Perceptron - PCA 2","MNIST")

print(82 * '_')
print("PCA - 10 Components")
print(82 * '_')
classifiers.nc_fit(orl_pca_train_10,orl_pca_test_10,orl_lables_train,orl_labels_test,"Nearest Centroid  - PCA 10","ORL")
classifiers.nc_fit(minst_pca_train_10,minst_pca_test_10,labels_training,labels_test,"Nearest Centroid  - PCA 10","MNIST")
classifiers.nn_fit(orl_pca_train_10,orl_pca_test_10,orl_lables_train,orl_labels_test,"Nearest Neighbor - PCA 10", "ORL")
classifiers.nn_fit(minst_pca_train_10,minst_pca_test_10,labels_training,labels_test,"NN - PCA 10","MNIST")
classifiers.nsc_fit(minst_pca_train_10,labels_training,minst_pca_test_10,labels_test,2,10,0,"NSC - 2 Clusters - PCA 10","MNIST")
classifiers.nsc_fit(minst_pca_train_10,labels_training,minst_pca_test_10,labels_test,3,10,0,"NSC - 3 Clusters - PCA 10","MNIST")
classifiers.nsc_fit(minst_pca_train_10,labels_training,minst_pca_test_10,labels_test,5,10,0,"NSC - 5 Clusters - PCA 10","MNIST")
classifiers.nsc_fit(orl_pca_train_10,orl_lables_train,orl_pca_test_10,orl_labels_test,2,40,1,"NSC - 2 Clusters - PCA 10","ORL")
classifiers.nsc_fit(orl_pca_train_10,orl_lables_train,orl_pca_test_10,orl_labels_test,3,40,1,"NSC - 3 Clusters - PCA 10","ORL")
classifiers.nsc_fit(orl_pca_train_10,orl_lables_train,orl_pca_test_10,orl_labels_test,5,40,1,"NSC - 5 Clusters - PCA 10","ORL")
classifiers.batch_perceptron_train_and_classify(orl_pca_train_10,orl_lables_train,40,"Batch Perceptron - PCA 2","ORL")
classifiers.batch_perceptron_train_and_classify(minst_pca_train_10,labels_training,10,"Batch Perceptron - PCA 2","MNIST")
classifiers.mse_perceptron_train_and_classify(orl_pca_train_10,orl_lables_train,40,"MSE Perceptron - PCA 2","ORL")
classifiers.mse_perceptron_train_and_classify(minst_pca_train_10,labels_training,10,"MSE Perceptron - PCA 2","MNIST")

'''
print('--------------------- Plots---------------------')
#plot.plot_average_mnist(images_training,labels_training)

#nc_pca_predicted_lables = classifiers.nc_fit(minst_pca_train,minst_pca_test,labels_training,labels_test,"Nearest Centroid - PCA 2","MNIST")
#nc_pca_orl_predicted_lables = classifiers.nc_fit(orl_pca_train,orl_pca_test,orl_lables_train,orl_labels_test,"Nearest Centroid - PCA 2","MNIST")

#nsc_pca_predicted_lables = classifiers.nsc_fit(minst_pca_train,labels_training,minst_pca_test,labels_test,2,10,0,"NSC - 2 Clusters - PCA 2","MNIST")
#nsc_pca_predicted_lables = classifiers.nsc_fit(minst_pca_train,labels_training,minst_pca_test,labels_test,5,10,0,"NSC - 5 Clusters - PCA 2","MNIST")
#plot.plot_data(minst_pca_test,labels_test,"MNIST PCA Test Data",1)
#plot.plot_data(minst_pca_test,nc_pca_predicted_lables,"Nearest Centroid Classified MINST Test Data",2)

#plot.plot_confusion_matrix(orl_labels_test,nc_pca_orl_predicted_lables,"Confusion Matrix NC ORL PCA test")
#plot.plot_data(minst_pca_test,nsc_pca_predicted_lables,"Nearest Sub Centroid - 3 clusters, Classified MINST Test Data",2)


#plot.plot_data(minst_pca_test,labels_test,"NN MINST Test Data",1)
#plot.plot_data(minst_pca_test,minst_nn_pca_pred_labels2,"NN Classified MINST Test Data",2)

#plot.plot_data(orl_pca_test,orl_labels_test,"NN ORL Test Data")
#plot.plot_data(orl_pca_test,nn_orl_pca_pred_labels2,"NN Classified ORL Test Data")
'''