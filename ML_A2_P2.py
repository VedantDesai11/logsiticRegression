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


class Model:
    def __init__(self, lr=0.01, batch_size=32, tolerance=0.0001):

        self.learning_rate = lr
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.iterations = 0
        self.actual = None

    def train(self, train, label):

        train_idx = np.arange(len(label))
        np.random.shuffle(train_idx)
        batches = np.array_split(train_idx, len(label)//self.batch_size+1)
        W = np.random.random((28,28))
        bias = 1
        m = self.batch_size

        oldloss = 20

        for i in range(10000):
            for batch in batches:

                X = train[batch]
                l = label[batch].reshape((len(batch), 1))

                z = np.dot(X, W.T) + bias

                h = softmax(z)
                dz = h - l
                dW = 1 / m * np.dot(dz.T, X)
                db = np.sum(dz) / m

                W = W - self.learning_rate * dW
                bias -= self.learning_rate * db

                newloss = neg_log_loss(h, l)

            # check threshold
            if newloss - oldloss < self.tolerance:
                break

        self.W = W
        self.bias = bias


    def predict(self, test, threshold=0.5):

        X = test[:, [0, 1]]
        testsize = X.shape[0]

        W = self.W
        b = self.bias
        z = np.dot(X, W.T) + b
        h = sigmoid(z)

        predictions = np.zeros((testsize,1))
        class1idx = np.where(h > threshold)
        class0idx = np.where(h <= threshold)
        predictions[class1idx[0]] = 1
        predictions[class0idx[0]] = 0

        actual = test[:, 2].reshape((testsize,1))
        accuracy = (np.count_nonzero(np.equal(actual, predictions))/testsize) * 100
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

    def calculateAUC(self, x, y):
        area = 0
        for i in range(len(x)):
            if i != 0:
                area += (x[i] - x[i - 1]) * y[i - 1] + (0.5 * (x[i] - x[i - 1]) * (y[i] - y[i - 1]))

        return -area


    def ROC(self, test):

        X = test[:, [0, 1]]
        testsize = X.shape[0]

        W = self.W
        b = self.bias
        z = np.dot(X, W.T) + b
        h = sigmoid(z)

        predictions = np.zeros((testsize, 1))
        actual = test[:, 2].reshape((testsize, 1))

        truePositiveRate = []
        falsePositiveRate = []

        for i in range(0, 100, 5):
            threshold = i/100
            class1idx = np.where(h > threshold)
            class0idx = np.where(h <= threshold)
            predictions[class1idx[0]] = 1
            predictions[class0idx[0]] = 0

            TP, FP, TN, FN = self.confusionMatrix(actual, predictions)

            truePositiveRate.append(TP/500)
            falsePositiveRate.append(FP/500)

        truePositiveRate.append(0)
        falsePositiveRate.append(0)
        auc = self.calculateAUC(falsePositiveRate, truePositiveRate)

        plt.plot(falsePositiveRate, truePositiveRate)
        plt.title(f'ROC Curve, AUC = {auc}')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()


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




