import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import math
from copy import deepcopy


def neg_log_loss(pred, label):
    loss = -np.log(pred[int(label)])
    return loss

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


X_train, X_test, y_train, y_test = idx2numpy.convert_from_file('mnist/train-images-idx3-ubyte'), \
                                       idx2numpy.convert_from_file('mnist/t10k-images-idx3-ubyte'), \
                                       idx2numpy.convert_from_file('mnist/train-labels-idx1-ubyte'), \
                                       idx2numpy.convert_from_file('mnist/t10k-labels-idx1-ubyte')

train_idx = np.where(y_train < 5)[0]
test_idx = np.where(y_test < 5)[0]

X_train, y_train, X_test, y_test = X_train[train_idx], y_train[train_idx], X_test[test_idx], y_test[test_idx]

# Normalize
X_train = X_train/255.0
X_test = X_test/255.0

a = np.arange(0,32)

X = X_train[a]
y = y_train[a].reshape((32, 1))


W = np.random.random((28*28, 5))

bias = 1

dW = 0
db = 0
a = X[0].flatten()
X = X.reshape((32,28*28))
if np.array_equal(X[0], a):
    print('ccc')
exit()
E = np.zeros((32, 5))  # (5x1)

for i in range(len(y)):
    E[i][y[i]] = 1

z = np.dot(X, W) + bias
h = softmax(z)
dz = h - E
dW = np.dot(X.T, dz) / 32
db = np.sum(dz)/32


# for i, data in enumerate(X):
#     x = data.reshape(28*28, 1)
#     E = np.zeros((5, 1))  # (10*1)
#     E[y[i]][0] = 1
#     z = np.dot(W.T, x) + bias
#     h = softmax(z).reshape(5,1)
#     dz = h - E
#     dW = np.dot(x, dz.T)
#     db = np.sum(dz)


# https://github.com/kmdanielduan/Logistic-Regression-on-MNIST-with-NumPy-from-Scratch/blob/master/logistic_regression.py
dW = dW/32
db = db/32

W = W - 0.001 * dW
bias = bias - 0.001 * db

#newloss = neg_log_loss(h, y)



