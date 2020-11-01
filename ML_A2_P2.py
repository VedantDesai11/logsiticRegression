import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import math
from copy import deepcopy

def cross_entropy(p, q):
	return -sum([p[i]*math.log(q[i]) for i in range(len(p))])


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class Model:
    def __init__(self, lr=0.01, batch_size=32, tolerance=0.0001):

        self.learning_rate = lr
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.iterations = 0
        self.actual = None

    def train(self, train, label):

        # number of different classes to classify. 5 in our case
        classes = len(np.unique(label))

        batch_size = self.batch_size

        # training indices from 0 - len(train_set)
        train_idx = np.arange(len(label))

        # shuffle training indices
        np.random.shuffle(train_idx)

        # create batches of given batch size from shuffled indices
        batches = np.array_split(train_idx, len(label)//batch_size+1)

        # Initialize weight matrix, bias and set batch_size
        W = np.random.random((28*28, classes))
        bias = 1

        for i in range(10000):
            for batch in batches:

                X = np.zeros((batch_size, 28*28))
                for j in range(len(train[batch])):
                    X[j] = train[batch][j].flatten()

                l = label[batch]
                y = np.zeros((batch_size, classes))  # Shape = (32,5)

                for j in range(len(y)):
                    y[j][l[j]] = 1

                z = np.dot(X, W) + bias # Shape = (32,5)
                h = softmax(z) # Shape = (32,5)
                dz = h - y # Shape = (32,5)
                dW = np.dot(X.T, dz) / batch_size # Shape = (32,764)
                db = np.sum(dz) / batch_size

                if i % 1000 == 0:
                    print(h[-1], y[-1])
                    loss = cross_entropy(h[-1].reshape(classes,1), y[-1].reshape(classes,1))
                    print(f"Loss at iteration({i} = {loss})")

                W = W - self.learning_rate * dW
                bias -= self.learning_rate * db

        self.W = W
        self.bias = bias


    def predict(self, test, l):

        # number of different classes to classify. 5 in our case
        classes = len(np.unique(l))
        testsize = test.shape[0]

        # get trained weights
        W = self.W
        bias = self.bias

        X = test.reshape((testsize, 28 * 28))  # Shape = (test_len,764)

        z = np.dot(X, W) + bias
        h = softmax(z)

        predictions = np.zeros((testsize,1))

        for i in range(len(h)):
            predictions[i] = np.argmax(h[i])

        accuracy = (np.count_nonzero(np.equal(l, predictions))/testsize) * 100
        print(f'Accuracy with LR({self.learning_rate}) = {accuracy}%')

        return predictions


    def confusionMatrix(self, actual, predicted):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(predicted)):
            if actual[i] == predicted[i] == 1:
                TP += 1
            if predicted[i] == 1 and actual[i] != predicted[i]:
                FP += 1
            if actual[i] == predicted[i] == 0:
                TN += 1
            if predicted[i] == 0 and actual[i] != predicted[i]:
                FN += 1

        return TP, FP, TN, FN



if __name__ == "__main__":

    batchsize = 32
    tolerance = 0.0001
    learningRate = 0.001

    X_train, X_test, y_train, y_test = idx2numpy.convert_from_file('mnist/train-images-idx3-ubyte'), \
                                       idx2numpy.convert_from_file('mnist/t10k-images-idx3-ubyte'), \
                                       idx2numpy.convert_from_file('mnist/train-labels-idx1-ubyte'), \
                                       idx2numpy.convert_from_file('mnist/t10k-labels-idx1-ubyte')

    train_subset_idx = np.where(y_train < 5)[0]
    test_subset_idx = np.where(y_test < 5)[0]

    X_train, y_train, X_test, y_test = X_train[train_subset_idx], y_train[train_subset_idx], X_test[test_subset_idx], y_test[test_subset_idx]

    # Normalize
    X_train = X_train / 255.0
    X_test = X_test / 255.0



    model = Model(learningRate, batchsize, tolerance)
    model.train(X_train, y_train)
    predictions = model.predict(X_test, y_test)
    print(predictions)




