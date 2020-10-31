import numpy as np
import matplotlib.pyplot as plt
import math
from copy import deepcopy


def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def featureDistance(W):

    sum = 0
    for weight in W:
        sum += weight*weight

    return math.sqrt(sum)


class Dataset:
    def __init__(self, trainsize, testsize):
        self.train = self.generateData(trainsize)
        self.test = self.generateData(testsize)

    def generateData(self, N):

        dataset = np.zeros((N, 3))
        classSize = N // 2
        mu1 = [1, 0]
        mu2 = [0, 1.5]
        sigma1 = np.matrix('1 0.75; 0.75 1')
        sigma2 = np.matrix('1 0.75; 0.75 1')

        data = np.concatenate((np.random.multivariate_normal(mu1, sigma1, classSize), np.random.multivariate_normal(mu2, sigma2, classSize)))
        for i, point in enumerate(data):
            dataset[i][0], dataset[i][1] = point[0], point[1]
            if i >= classSize:
                dataset[i][2] = 1

        return dataset


class Model:
    def __init__(self, lr=0.01, batch_size=32, tolerance=0.0001):

        self.learning_rate = lr
        self.batch_size = batch_size
        self.tolerance = tolerance
        train_idx = np.arange(len(dataset.train))
        np.random.shuffle(train_idx)
        self.batches = np.array_split(train_idx, 1000//batch_size+1)
        self.W = np.zeros((1, 2))
        self.bias = 1
        self.iterations = 0
        self.actual = None

    def train(self, train):

        W = self.W
        bias = self.bias
        m = self.batch_size

        oldloss = 20

        for i in range(10000):

            oldW = deepcopy(W)
            oldb = deepcopy(bias)

            for batch in self.batches:
                data = train[batch]

                X = data[:,[0, 1]]
                print(f'X.shape = {X.shape}')
                l = data[:,2].reshape((len(batch), 1))
                print(f'l.shape = {l.shape}')
                z = np.dot(X, W.T) + bias
                print(f'z.shape = {z.shape}')
                h = sigmoid(z)
                print(f'h.shape = {h.shape}')
                dz = h - l
                print(f'dz.shape = {dz.shape}')

                dW = 1 / m * np.dot(dz.T, X)
                print(f'dW.shape = {dW.shape}')
                print(W)
                db = np.sum(dz) / m
                print(f'db.shape = {db.shape}')

                W = W - self.learning_rate * dW
                bias -= self.learning_rate * db

                newloss = loss(h, l)

            # check threshold
            if self.learning_rate * featureDistance(W[0]) < self.tolerance and newloss - oldloss < self.tolerance:
                break

        self.W = W
        self.bias = bias
        self.num_iterations = i


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

    train_size = 1000
    test_size = 1000
    batchsize = 32
    tolerance = 0.0001
    learningRates = [1, 0.1, 0.01, 0.001, 0.0001]

    dataset = Dataset(train_size, test_size)

    model = Model(learningRates[2], batchsize, tolerance)
    model.train(dataset.train)
    predictions = model.predict(dataset.test)
    model.ROC(dataset.test)

    iterations = []
    for lr in learningRates:
        model = Model(lr, batchsize, tolerance)
        model.train(dataset.train)
        iterations.append(model.num_iterations)

    plt.plot(learningRates, iterations)
    plt.show()


