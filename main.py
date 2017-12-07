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
'''
print(82 * '_')
print("Classifiers")
print(82 * '_')
print('name\t\t\t\ttime\tscore\t\tdata')

classifiers.nc_fit(orl_data_train,orl_data_test,orl_lables_train,orl_labels_test, "Nearest Centroid", "ORL")
classifiers.nc_fit(images_training,images_test,labels_training,labels_test,"Nearest Centroid","MNIST")
#classifiers.nn_fit(orl_data_train,orl_data_test,orl_lables_train,orl_labels_test,"Nearest Neighbor","ORL")
#classifiers.nn_fit(images_training,images_test,labels_training,labels_test, "NN","MNIST")
classifiers.nsc_fit(images_training,labels_training,images_test,labels_test,2,10,"NSC - 2 Clusters","MNIST")
#classifiers.nsc_fit(orl_data_train,orl_data_test,orl_lables_train,orl_labels_test,2,10,"NSC - 2 Clusters","ORL")
classifiers.nsc_fit(images_training,labels_training,images_test,labels_test,3,10,"NSC - 3 Clusters","MNIST")
classifiers.nsc_fit(images_training,labels_training,images_test,labels_test,5,10,"NSC - 5 Clusters","MNIST")

print(82 * '_')
print("PCA - 2 Components")
print(82 * '_')
classifiers.nc_fit(orl_pca_train,orl_pca_test,orl_lables_train,orl_labels_test, "Nearest Centroid - PCA 2","ORL")
classifiers.nc_fit(minst_pca_train,minst_pca_test,labels_training,labels_test,"Nearest Centroid - PCA 2","MNIST")
#classifiers.nn_fit(orl_pca_train,orl_pca_test,orl_lables_train,orl_labels_test,"Nearest Neighbor - PCA 2","ORL")
#classifiers.nn_fit(minst_pca_train,minst_pca_test,labels_training,labels_test, "NN - PCA 2 ","MNIST")
classifiers.nsc_fit(minst_pca_train,labels_training,minst_pca_test,labels_test,2,10,"NSC - 2 Clusters - PCA 2","MNIST")
classifiers.nsc_fit(minst_pca_train,labels_training,minst_pca_test,labels_test,3,10,"NSC - 3 Clusters - PCA 2","MNIST")
classifiers.nsc_fit(minst_pca_train,labels_training,minst_pca_test,labels_test,5,10,"NSC - 5 Clusters - PCA 2","MNIST")

print(82 * '_')
print("PCA - 10 Components")
print(82 * '_')
classifiers.nc_fit(orl_pca_train_10,orl_pca_test_10,orl_lables_train,orl_labels_test,"Nearest Centroid  - PCA 10","ORL")
classifiers.nc_fit(minst_pca_train_10,minst_pca_test_10,labels_training,labels_test,"Nearest Centroid  - PCA 10","MNIST")
#classifiers.nn_fit(orl_pca_train_10,orl_pca_test_10,orl_lables_train,orl_labels_test,"Nearest Neighbor - PCA 10", "ORL")
#classifiers.nn_fit(minst_pca_train_10,minst_pca_test_10,labels_training,labels_test,"NN - PCA 10","MNIST")
classifiers.nsc_fit(minst_pca_train,labels_training,minst_pca_test,labels_test,2,10,"NSC - 2 Clusters - PCA 10","MNIST")
classifiers.nsc_fit(minst_pca_train,labels_training,minst_pca_test,labels_test,3,10,"NSC - 3 Clusters - PCA 10","MNIST")
classifiers.nsc_fit(minst_pca_train,labels_training,minst_pca_test,labels_test,5,10,"NSC - 5 Clusters - PCA 10","MNIST")

'''


##---------------------Nearest Centroid----------------------------------------


#plot.plot_data(orl_pca_test,orl_labels_test,"NC ORL Test Data")
#plot.plot_data(orl_pca_test,orl_pca_pred_labels2,"NC Classified ORL Test Data")
#plot.plot_orl_n_by_m(orl_data_train,10,28)
#plot.plot_orl_face(orl_data_train)

#plot.plot_data(minst_pca_test,labels_test,"NC MINST Test Data",1)
#plot.plot_data(minst_pca_test,minst_pca_pred_labels2,"NC Classified MINST Test Data",2)

#print(images_test[4])
#plot.plot_mnist_n_by_m(images_test,15,15)
#plot.plot_images_separately(images_test)
#plot.plot_2_and_1(images_test)
#test =  images_test[0:100]

#plot.plot_10_by_10_images(test)
#plot.plot_mnist_digit(images_test[4])

##print(str(labels_test[4]))

#print('--------------------- Nearest Neighbor - ORL ---------------------')


#plot.plot_data(orl_pca_test,orl_labels_test,"NN ORL Test Data")
#plot.plot_data(orl_pca_test,nn_orl_pca_pred_labels2,"NN Classified ORL Test Data")
'''
print('--------------------- Nearest Neighbor - MNIST ---------------------') 
#print('Score from function ' + str(minst_score_nn))
print('Score from PCA - 2 Components ' + str(minst_pca_score_2_nn))
print('Score from PCA - 10 Components ' + str(minst_pca_score_10_nn))

plot.plot_data(minst_pca_test,labels_test,"NN MINST Test Data",1)
plot.plot_data(minst_pca_test,minst_nn_pca_pred_labels2,"NN Classified MINST Test Data",2)

print('--------------------- Nearest Sub Class - ORL ---------------------')

orl_nsc_train = orl_data_train #.transpose()
orl_nsc_test = orl_data_test #.transpose()


nsc_score2_orl= classifiers.nsc_fit(orl_nsc_train,orl_nsc_test,orl_lables_train,orl_labels_test,2,40)
#nsc_score3_orl = classifiers.nsc_fit(orl_data_train,orl_data_test,orl_lables_train,orl_labels_test,3,40)
#nsc_score5_orl = classifiers.nsc_fit(orl_data_train,orl_data_test,orl_lables_train,orl_labels_test,5,40)

print('Score from function NSC - 2 clusters ' + str(nsc_score2_orl))
#print('Score from function NSC - 3 clusters ' + str(nsc_score3_orl))
#print('Score from function NSC - 5 clusters ' + str(nsc_score5_orl))


print('--------------------- Nearest Sub Class - MNIST ---------------------')
nsc_score2_mnist= classifiers.nsc_fit(images_training,labels_training,images_test,labels_test,2,10)
nsc_score3_mnist = classifiers.nsc_fit(images_training,labels_training,images_test,labels_test,3,10)
nsc_score5_mnist = classifiers.nsc_fit(images_training,labels_training,images_test,labels_test,5,10)

print('Score from function NSC - 2 clusters ' + str(nsc_score2_mnist))
print('Score from function NSC - 3 clusters ' + str(nsc_score3_mnist))
print('Score from function NSC - 5 clusters ' + str(nsc_score5_mnist))
'''

print('--------------------- Batch Perceptron ---------------------')

#new_lbls = classifiers.binary_classifyer(labels_training,2)
#print(new_lbls)

weight_array = []
for i in range(0,10):
    new_lbls = classifiers.binary_classifyer(labels_training,i)
    weight_array.append(classifiers.perceptron_sgd_plot(images_training,new_lbls,i))
   #np.dot(w, xtest[i])
#print(weight_array)

score = classifiers.predict(weight_array,images_test,labels_test)
print(score)
#     labels.append(str(i))

#     for k in range(0, n_clust):
#         centers.append(kmeans.cluster_centers_[k, :])

# for i in range(0, 10):
#     weight_array.append(classifiers.perceptron_sgd_plot(X,y))

# data = Xtrain[np.nonzero(Xtrain_lbls == 0)]

#print(classifiers.perceptron_sgd_plot(X,y))