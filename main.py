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

X_train, X_test, Y_train, Y_test = train_test_split(orl_data,orl_labels,test_size=0.30,random_state=42)

clf = NearestCentroid()
# Train with the data
clf.fit(X_train,Y_train)

# Create predition for train data
y_pred_test = clf.predict(X_test)

#How well does it fit
score = clf.score(X_test,Y_test)

print(score)

print('---------------------MINST - Nearest Centroid---------------------')
mndata = MNIST('./')
images, labels = mndata.load_training()
