import numpy as np
import scipy.io as sio
from mnist import MNIST
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split

mndata = MNIST('./')
images, labels = mndata.load_training()


orl_data = np.array(sio.loadmat('orl_data.mat')['data']).transpose()
orl_labels = np.array(sio.loadmat('orl_lbls.mat')['lbls']).ravel()


X_train, X_test = train_test_split(orl_data, random_state=42)

clf = NearestCentroid()
clf.fit(orl_data, orl_labels)

#Z = clf.predict([orl_data.ravel(), orl_labels])
y_pred = clf.predict(orl_data)
print(y_pred)

diff = np.mean(orl_labels-y_pred)
print(diff)

