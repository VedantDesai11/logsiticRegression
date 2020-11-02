import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm


def cross_entropy(q, p):
    return -sum([p[j] * math.log(q[j]) for j in range(len(p))])


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class Model:
    def __init__(self, lr=0.01, batch_size=32, tolerance=0.0001):

        self.learning_rate = lr
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.iterations = 0
        self.actual = None

    def train(self, train, label, classes=5):

        batch_size = self.batch_size
        train_size = train.shape[0]

        # training indices from 0 - len(train_set)
        train_idx = np.arange(train_size)

        # shuffle training indices
        np.random.shuffle(train_idx)

        # create batches of given batch size from shuffled indices
        batches = np.array_split(train_idx, train_size // batch_size + 1)

        # Initialize weight matrix, bias and set batch_size
        W = np.random.random((classes, 28*28)) * 0.01
        bias = 1
        loss_list = []

        for i in tqdm(range(10000)):
            for batch in batches:
                X = train[batch]
                y = label[batch]

                z = np.dot(X, W.T) + bias  # Shape = (32,5)

                h = np.zeros((len(batch), classes))

                for j,val in enumerate(z):
                    h[j] = softmax(val)

                dz = h - y  # Shape = (32,5)
                dW = np.dot(dz.T, X) / len(batch)  # Shape = (5,764)
                db = np.sum(dz)/len(batch)

            loss = cross_entropy(h[-1].reshape(classes, 1), y[-1].reshape(classes, 1))[0]
            loss_list.append(loss)

            if loss < 0.05:
                print(f"Performing Early Stopping.\nLoss at iteration {i} = {loss}")
                break

            if i % 500 == 0:
                print(f"\nLoss at iteration {i} = {loss}")

            W = W - self.learning_rate * dW
            bias = bias - self.learning_rate * db

        self.W = W
        self.bias = bias
        self.losslist = loss_list

    def predict(self, test, l):

        # number of different classes to classify. 5 in our case
        classes = len(np.unique(l))
        testsize = test.shape[0]
        actual = l.reshape((testsize, 1))

        # get trained weights
        W = self.W
        bias = self.bias

        X = np.array([X.flatten() for X in test])

        z = np.dot(X, W.T) + bias
        predictions = np.zeros((testsize, 1))
        correct = 0

        for i, val in enumerate(z):
            prediction = np.argmax(softmax(val))
            predictions[i] = prediction
            if prediction == actual[i]:
                correct += 1

        accuracy = (correct/testsize) * 100
        print(f'Accuracy = {accuracy}%')

        return predictions


    def precision_Recall(self, actual, predicted):

        PrecisionList = []
        RecallList = []

        confusionMatrix = np.zeros((5,5))

        for i in range(len(predicted)):
            pred = int(predicted[i])
            a = actual[i]

            if pred == a:
                confusionMatrix[pred][pred] += 1
            else:
                confusionMatrix[pred][a] += 1

        for c in np.unique(actual):
            TP = confusionMatrix[c][c]
            TN = 0
            FP = 0
            FN = 0
            for row in np.unique(actual):
                for column in np.unique(actual):
                    if column != c and row != c:
                        TN += confusionMatrix[row][column]
                    if row == c and column != c:
                        FP += confusionMatrix[row][column]
                    if row != c and column == c:
                        FN += confusionMatrix[row][column]

            RecallList.append(TP/(TP+FN))
            PrecisionList.append(TP/(TP+FP))

        return confusionMatrix, PrecisionList, RecallList



if __name__ == "__main__":
    batchsize = 200
    tolerance = 0.0001
    learningRate = 0.01

    X_train, X_test, y_train, y_test = idx2numpy.convert_from_file('mnist/train-images-idx3-ubyte'), \
                                       idx2numpy.convert_from_file('mnist/t10k-images-idx3-ubyte'), \
                                       idx2numpy.convert_from_file('mnist/train-labels-idx1-ubyte'), \
                                       idx2numpy.convert_from_file('mnist/t10k-labels-idx1-ubyte')

    train_subset_idx = np.where(y_train < 5)[0]
    test_subset_idx = np.where(y_test < 5)[0]

    X_train, y_train, X_test, y_test = X_train[train_subset_idx], y_train[train_subset_idx], X_test[test_subset_idx], \
                                       y_test[test_subset_idx]

    # Normalize
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # flatten image data from 28x28 to 784,1
    X_train = np.array([X.flatten() for X in X_train])

    print(f'X_train.shape={X_train.shape}' )

    # one hot encoding for labels
    one_hot_y_train = np.zeros((len(y_train), len(np.unique(y_train))))  # Shape = (32,5)

    for i in range(len(one_hot_y_train)):
        one_hot_y_train[i][y_train[i]] = 1

    print(f'one_hot_y_train.shape={one_hot_y_train.shape}')

    model = Model(learningRate, batchsize, tolerance)
    model.train(X_train, one_hot_y_train)

    predictions = model.predict(X_test, y_test)
    cm, precision, recall = model.precision_Recall(y_test, predictions)

    print(cm)
    for i, c in enumerate(np.unique(y_test)):
        print(f'For Class {c}, Precision = {precision[i]}, Recall = {recall[i]}')

    plt.plot(range(len(model.losslist)), model.losslist)
    plt.show()



