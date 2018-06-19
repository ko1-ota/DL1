# coding:utf-8

from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer as lb

import numpy as np


def homework(train_X, train_y, test_X):
    np.random.seed(1234)

    npexp = np.exp
    npsqrt = np.sqrt
    npsum = np.sum
    npdot = np.dot
    npnewaxis = np.newaxis
    npones = np.ones

    def softmax(x):
        return npexp(x) / npsum(npexp(x), axis=1, keepdims=True)
    def d_softmax(x):
        return softmax(x) * (1 - softmax(x))

    UNITS_IN = len(train_X[0])
    UNITS_OUT = 10
    UNITS_HIDDEN = 30
    nprandomuniform = np.random.uniform
    W1 = nprandomuniform(low=-npsqrt(6. / (UNITS_IN + UNITS_HIDDEN)), high=npsqrt(6. / (UNITS_IN + UNITS_HIDDEN)), size=(UNITS_IN, UNITS_HIDDEN)).astype('float32')
    b1 = np.zeros(UNITS_HIDDEN).astype('float32')
    W2 = nprandomuniform(low=-npsqrt(6. / (UNITS_HIDDEN + UNITS_OUT)), high=npsqrt(6. / (UNITS_HIDDEN + UNITS_OUT)), size=(UNITS_HIDDEN, UNITS_OUT)).astype('float32')
    b2 = np.zeros(UNITS_OUT).astype('float32')

    ook = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    train_y = lb().fit(ook).transform(train_y)
    train_X_, valid_X, train_y_, valid_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

    def train(X, t):
        u1 = npdot(X, W1) + b1
        z1 = softmax(u1)

        u2 = npdot(z1, W2) + b2
        z2 = softmax(u2)

        y = z2
        cost = npsum(npsum(-t[:, npnewaxis] * np.log(z2)))
        delta2 = y - t[:, npnewaxis]
        delta1 = d_softmax(u1) * npdot(delta2, W2.T)

        EPS = 0.01

        dW2 = npdot(z1.T, delta2)
        db2 = delta2

        W2 = W2 - EPS * dW2
        b2 = b2 - EPS * db2

        dW1 = npdot(X.T, delta1)
        db1 = delta1

        W1 = W1 - EPS * dW1
        b1 = b1 - EPS * db1

    for epoch in range(2000):
        for X, t in zip(train_X, train_y):
            # train(x[npnewaxis, :], t)

            X = X[npnewaxis, :]
            t = t[npnewaxis, :]

            u1 = npdot(X, W1) + b1
            z1 = u1

            u2 = npdot(z1, W2) + b2
            z2 = softmax(u2)

            y = z2
            cost = npsum(npsum(-t * np.log(z2)))
            delta2 = y - t
            delta1 = d_softmax(u1) * npdot(delta2, W2.T)

            EPS = 0.1

            dW2 = npdot(z1.T, delta2)
            db2 = delta2

            W2 = W2 - EPS * dW2
            b2 = b2 - EPS * db2

            dW1 = npdot(X.T, delta1)
            db1 = delta1

            W1 = W1 - EPS * dW1
            b1 = b1 - EPS * db1
        
        u1 = npdot(test_X, W1) + b1
        z1 = u1

        u2 = npdot(z1, W2) + b2
        z2 = softmax(u2)

        y = z2.argmax(axis=1)
        print y, epoch
    return y

def load_mnist():
    mnist = fetch_mldata('MNIST original', data_home=".")
    mnist_X, mnist_y = shuffle(mnist.data.astype('float32'),
                               mnist.target.astype('int32'), random_state=42)

    mnist_X = mnist_X / 255.0

    return train_test_split(mnist_X, mnist_y,
                test_size=0.2,
                random_state=42)

def validate_homework():
    train_X, test_X, train_y, test_y = load_mnist()

    # validate for small dataset
    train_X_mini = train_X[:100]
    train_y_mini = train_y[:100]
    test_X_mini = test_X[:100]
    test_y_mini = test_y[:100]

    pred_y = homework(train_X_mini, train_y_mini, test_X_mini)
    # print test_y_mini
    print float([t==p for t, p in zip(test_y_mini, pred_y)].count(True)) / float(len(test_y_mini))
    print(f1_score(test_y_mini, pred_y, average='macro'))

def score_homework():
    train_X, test_X, train_y, test_y = load_mnist()
    pred_y = homework(train_X, train_y, test_X)
    print(f1_score(test_y, pred_y, average='macro'))


validate_homework()