import numpy as np
import matplotlib.pyplot as plt
import math
from copy import deepcopy


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def generateData(size):

    mu1 = [1, 0]
    mu2 = [0, 1.5]
    sigma1 = np.matrix('1 0.75; 0.75 1')
    sigma2 = np.matrix('1 0.75; 0.75 1')

    x, y = np.random.multivariate_normal(mu1, sigma1, size).T  # LABEL 0
    label0 = np.zeros((size,1))
    a, b = np.random.multivariate_normal(mu2, sigma2, size).T  # LABEL 1
    label1 = np.ones((size,1))

    x1 = np.concatenate((x, a))
    y1 = np.concatenate((y, b))
    label = np.concatenate((label0, label1))

    return np.array((x1, y1)).T, label


class Dataset:
    def __init__(self, trainsize, testsize):
        trainsize = trainsize//4
        testsize = testsize//4
        mu1 = [1, 0]
        mu2 = [0, 1.5]
        sigma1 = np.matrix('1 0.75; 0.75 1')
        sigma2 = np.matrix('1 0.75; 0.75 1')

        traindata0 = np.concatenate((np.random.multivariate_normal(mu1, sigma1, trainsize),
                            np.random.multivariate_normal(mu1, sigma1, trainsize)))
        traindata1 = np.concatenate((np.random.multivariate_normal(mu1, sigma1, trainsize),
                            np.random.multivariate_normal(mu1, sigma1, trainsize)))
        testdata0 = np.concatenate((np.random.multivariate_normal(mu1, sigma1, testsize),
                                     np.random.multivariate_normal(mu1, sigma1, testsize)))
        testdata1 = np.concatenate((np.random.multivariate_normal(mu1, sigma1, testsize),
                                     np.random.multivariate_normal(mu1, sigma1, testsize)))

        label0 = np.concatenate((np.zeros((trainsize, 1)),
                            np.ones((trainsize, 1))))
        label1 = np.concatenate((np.zeros((trainsize, 1)),
                                 np.ones((trainsize, 1))))

        train0 = np.append(traindata0, label0, 1)
        train1 = np.append(traindata1, label1, 1)
        test0 = np.append(testdata0, label0, 1)
        test1 = np.append(testdata1, label1, 1)

        self.train = np.concatenate((train0, train1))
        self.test = np.concatenate((test0, test1))


class Model:
    def __init__(self, lr, batch_size, tolerance):

        self.learning_rate = lr
        self.batch_size = batch_size
        self.tolerance = tolerance
        train_idx = np.arange(len(dataset.train))
        np.random.shuffle(train_idx)
        self.batches = np.array_split(train_idx, 1000//batch_size+1)
        self.W = np.zeros((1, 2))
        self.bias = 1
        self.iterations = 0

    def train(self):

        W = self.W
        bias = self.bias
        m = self.batch_size

        trainingLoss = []
        normGradient = []

        for i in range(10000):

            oldW = deepcopy(W)
            oldb = deepcopy(bias)

            for batch in self.batches:
                data = dataset.train[batch]
                X = data[:,[0, 1]]
                l = data[:,2].reshape((len(batch), 1))
                z = np.dot(X, W.T) + bias
                h = sigmoid(z)
                dz = h - l

                dW = 1 / m * np.dot(dz.T, X)
                db = np.sum(dz) / m

                W = W - self.learning_rate * dW
                bias -= self.learning_rate * db

            trainingLoss.append(loss(h, l))
            normGradient.append(self.learning_rate * featureDistance(W[0]))

            # check threshold
            if (abs(W - oldW) < self.tolerance).all() and abs(bias - oldb) < self.tolerance and trainingLoss[-1]-trainingLoss[-2] < self.tolerance:
                break

        self.W = W
        self.bias = bias
        self.iterations = i

        #plt.plot(range(len(normGradient)), normGradient)
        #plt.show()
        #plt.plot(range(len(trainingLoss)), trainingLoss)
        #plt.show()


def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def featureDistance(W):

    sum = 0
    for weight in W:
        sum += weight*weight

    return math.sqrt(sum)


def plotTrainDecisionBoundary(class0, class1, model):

    plt.scatter(class0[:, 0], class0[:, 1])
    plt.scatter(class1[:, 0], class1[:, 1])
    ax = plt.gca()
    ax.autoscale(False)
    x_vals = np.array(ax.get_xlim())
    y_vals = -(x_vals * model[0][0][0] + model[1]) / model[0][0][1]
    plt.plot(x_vals, y_vals, '--', c="red")
    plt.title("Testing Data with Decision Boundary.")
    plt.show()


def predict(model, Xtest, XTestlabel):

    z = np.dot(Xtest, model[0].T) + model[1]
    h = sigmoid(z)

    class0 = []
    class1 = []
    pred = []
    for i in range(len(h)):
        if h[i] > 0.5:
            pred.append(1)
            class1.append(Xtest[i])
        else:
            pred.append(0)
            class0.append(Xtest[i])

    class0 = np.array(class0)
    class1 = np.array(class1)

    count = 0
    for x in range(len(pred)):
        if pred[x] == XTestlabel[x]:
            count += 1

    plotTrainDecisionBoundary(class0, class1, model)

    print("Accuracy with LR="+str(learningRate)+": " + str((count / len(pred)) * 100) + "%")


def logisticRegression(Batch, datasetSize, learningRate, tolerance=0.0001, iterations=100000):

    # TRAINING DATA
    Xtrain, XTrainlabel = generateData(datasetSize)

    # TESTING DATA
    Xtest, XTestlabel = generateData(int(datasetSize / 2))

    if Batch == True:
        model = Training(Xtrain, XTrainlabel, 250, iterations, tolerance, learningRate)
    else:
        model = Training(Xtrain, XTrainlabel, 1, iterations, tolerance, learningRate)

    predict(model, Xtest, XTestlabel)


def Training(Xtrain, XTrainlabel, batchSize, iterations, tolerance, learningRate):

    Xtrain, XTrainlabel = unison_shuffled_copies(Xtrain, XTrainlabel)

    batches = np.split(Xtrain, int(Xtrain.shape[0]/batchSize))
    Labelbatches = np.split(XTrainlabel, int(XTrainlabel.shape[0]/batchSize))

    # weights initialization with 1
    W = np.ones((1, batches[0].shape[1]))
    # bias inialization with 0
    bias = 0

    m = batches[0].shape[0]
    trainingLoss = []
    normGradient = []
    batchesNumber = len(Xtrain)//batchSize

    for i in range(iterations//batchesNumber):
        oldW = deepcopy(W)
        oldb = deepcopy(bias)
        for batch in range(len(batches)):

            z = np.dot(batches[batch], W.T) + bias
            h = sigmoid(z)

            dz = h - Labelbatches[batch]

            if batchSize == 1:
                dW = np.dot(dz.T, batches[batch])
                db = np.sum(dz)
            else:
                dW = 1 / m * np.dot(dz.T, batches[batch])
                db = np.sum(dz) / m

            trainingLoss.append(loss(h, Labelbatches[batch]))
            normGradient.append(learningRate * featureDistance(W[0]))

            W = W - learningRate * dW
            bias -= learningRate * db

        # check threshold
        if (abs(W - oldW) < tolerance).all() and abs(bias - oldb) < tolerance:
            break

    print("Number of iterations: " + str(len(trainingLoss)))

    plt.subplot(1, 2, 1)  # two rows, one columns, first graph
    plt.plot(trainingLoss)
    plt.title('Training Loss')
    plt.xlabel("Number of Iterations")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)  # two rows, one columns, 2nd graph
    plt.plot(normGradient)
    plt.title('Norm Gradient')
    plt.xlabel("Number of Iterations")
    plt.ylabel("Gradient")
    plt.show()

    return W, bias


if __name__ == "__main__":

    dataSetSize = 1000

    train_size = 1000
    test_size = 1000

    dataset = Dataset(train_size, test_size)

    learningRates = [1, 0.1, 0.01, 0.001]
    learningRates.reverse()
    iterations = []
    for lr in learningRates:
        model = Model(lr, 32, 0.0001)
        model.train()
        iterations.append(model.iterations)

    learningRates.reverse()

    plt.plot(learningRates, iterations)
    plt.show()

    exit()
    # Batch Training
    for learningRate in learningRates:
        logisticRegression(True, dataSetSize, learningRate)

    # Online Training
    for learningRate in learningRates:
        logisticRegression(False, dataSetSize, learningRate)

