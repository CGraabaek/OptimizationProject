import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mnist import MNIST
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split
import nearest_centroid_classifiers

print('---------------------ORL - Nearest Centroid---------------------')
orl_data = np.array(sio.loadmat('orl_data.mat')['data']).transpose()
orl_labels = np.array(sio.loadmat('orl_lbls.mat')['lbls']).ravel()

orl_data_train, orl_data_test, orl_lables_train, orl_labels_test = train_test_split(orl_data,orl_labels,test_size=0.30,random_state=42)

orl_score = nearest_centroid_classifiers.nc_fit(orl_data_train,orl_data_test,orl_lables_train,orl_labels_test)

print('Score from function ' +str(orl_score))


print('---------------------MINST - Nearest Centroid---------------------')
mndata = MNIST('./', return_type='numpy')
images_training, labels_training = mndata.load_training()
images_test, labels_test = mndata.load_testing()

minst_score = nearest_centroid_classifiers.nc_fit(images_training,images_test,labels_training,labels_test)

print('Score from function ' +str(minst_score))
