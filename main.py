import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mnist import MNIST
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split

print('---------------------ORL - Nearest Centroid---------------------')
orl_data = np.array(sio.loadmat('orl_data.mat')['data']).transpose()
orl_labels = np.array(sio.loadmat('orl_lbls.mat')['lbls']).ravel()

orl_data_train, orl_data_test, orl_lables_train, orl_labels_test = train_test_split(orl_data,orl_labels,test_size=0.30,random_state=42)

clf = NearestCentroid()
# Train with the data
clf.fit(orl_data_train,orl_lables_train)

# Create predition for train data
y_pred_test = clf.predict(orl_data_test)

#How well does it fit
score = clf.score(orl_data_test,orl_labels_test)

print(score)

print('---------------------MINST - Nearest Centroid---------------------')
mndata = MNIST('./')
images_training, labels_training = mndata.load_training()
images_test, labels_test = mndata.load_testing()


img_train = mndata.process_images_to_numpy(images_training)
img_test = mndata.process_images_to_numpy(images_test)
lbl_train = mndata.process_images_to_numpy(labels_training)
lbl_test = mndata.process_images_to_numpy(labels_test)

# Train with the data
clf.fit(img_train,lbl_train)

# Create predition for train data
labels_pred_test = clf.predict(img_test)

#How well does it fit
score_minst = clf.score(img_test,lbl_test)
print(score_minst)
