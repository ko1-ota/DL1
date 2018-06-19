# coding:utf-8
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

import numpy as np


def homework(train_X, train_y, test_X):
    nparray = np.array
    norm = np.linalg.norm
    def dist(x, train_X):
        return [norm(tx- x) for tx in train_X]
    K = 20
    score = 0
    for k_ in range(1, K+1):
        train_X_, valid_X, train_y_, valid_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)
        dX = [dist(x, train_X_) for x in valid_X]
        knind = [list(nparray(x).argsort()[:k_]) for x in dX]
        kny = [list(train_y_[x]) for x in knind]
        county = [[[y, x.count(y)] for y in list(set(x))] for x in kny]
        pred_y = [nparray(x)[nparray(x)[:,1].argsort()[-1:]][0][0] for x in county]
        score_ = (valid_y==pred_y).tolist().count(True)
        if score_ > score:
            score = score_
            k = k_
    dX = [dist(x, train_X) for x in test_X]
    knind = [list(nparray(x).argsort()[:k]) for x in dX]
    kny = [list(train_y[x]) for x in knind]
    county = [[[y, x.count(y)] for y in list(set(x))] for x in kny
]    pred_y = [nparray(x)[nparray(x)[:,1].argsort()[-1:]][0][0] for x in county]
    return pred_y


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
    train_X_mini = train_X[:1000]
    train_y_mini = train_y[:1000]
    test_X_mini = test_X[:1000]
    test_y_mini = test_y[:1000]

    pred_y = homework(train_X_mini, train_y_mini, test_X_mini)
    print(f1_score(test_y_mini, pred_y, average='macro'))


# validate_homework()
# train_X, test_X, train_y, test_y = load_mnist()
# train_X_mini = train_X[:1000]
# train_y_mini = train_y[:1000]
# test_X_mini = test_X[:100]
# test_y_mini = test_y[:100]

# norm = np.linalg.norm
# def dist(x, train_X):
#     return [norm(tx - x) for tx in train_X]

# k = 5
# d = [dist(x, train_X_mini) for x in test_X_mini]

# nparray = np.array
# knind = [list(nparray(x).argsort()[:k]) for x in d]
# kny = [list(train_y_mini[x]) for x in knind]
# # county = [[[y, x.count(y)] for y in list(set(x))] for x in kny]
# # pred_y_mini = [nparray(x)[nparray(x)[:,1].argsort()[-1:]][0][0] for x in county]
# import collections
# pred_y_mini = [collections.Counter(x).most_common(1)[0][0] for x in kny]
# print((test_y_mini==pred_y_mini).tolist().count(True))

from benchmarker import Benchmarker
loop = 1000
with Benchmarker(loop, width=20) as bench:

    @bench('default')
    def default_vh(bm):
        validate_homework()