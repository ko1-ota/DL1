# coding:utf-8

from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer as lb

import numpy as np
import tensorflow as tf


def homework(train_X, train_y, test_X):
    tf.reset_default_graph()
    rnd = np.random.RandomState(1234)
    nprandomuniform = rnd.uniform

    UNITS_IN = len(train_X[0])
    UNITS_OUT = 10
    UNITS_HIDDEN = 20

    EPS = 0.01

    npnewaxis = np.newaxis
    npsqrt = np.sqrt
    npones = np.ones
    npzeros = np.zeros

    tfmatmul = tf.matmul
    tfsqrt = tf.sqrt
    tflog = tf.log
    tfsum = tf.reduce_sum
    tfgradients = tf.gradients
    sigmoid = tf.nn.sigmoid
    softmax = tf.nn.softmax


    X = tf.placeholder(tf.float32, [None, 784], name="X")
    t = tf.placeholder(tf.float32, [None, 10], name="t")

    W1 = tf.Variable(nprandomuniform(low=-npsqrt(6. / (UNITS_IN + UNITS_HIDDEN)), high=npsqrt(6. / (UNITS_IN + UNITS_HIDDEN)), size=(UNITS_IN, UNITS_HIDDEN)).astype('float32'), name="W1")
    b1 = tf.Variable(npzeros(UNITS_HIDDEN).astype('float32'), name="b1")
    W2 = tf.Variable(nprandomuniform(low=-npsqrt(6. / (UNITS_HIDDEN + UNITS_OUT)), high=npsqrt(6. / (UNITS_HIDDEN + UNITS_OUT)), size=(UNITS_HIDDEN, UNITS_OUT)).astype('float32'), name="W2")
    b2 = tf.Variable(npzeros(UNITS_OUT).astype('float32'), name="b2")
    params = [W1, b1, W2, b2]

    u1 = tfmatmul(X, W1) + b1
    z1 = u1
    u2 = tfmatmul(z1, W2) + b2
    y = softmax(u2)

    cost = -tfsum(tfsum(t * tflog(tf.clip_by_value(y, 1e-10, 1.0))))

    gW1, gb1, gW2, gb2 = tfgradients(cost, params)
    updates = [
        W1.assign_add(-EPS * gW1),
        b1.assign_add(-EPS * gb1),
        W2.assign_add(-EPS * gW2),
        b2.assign_add(-EPS * gb2)
    ]

    train = tf.group(*updates)

    valid = tf.argmax(y, axis=1)

    p_y = tf.placeholder(tf.float32, [None, 10])
    rate = [p_y==t].count(True) / tf.shape(p_y)[0]

    ook = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    train_y = lb().fit(ook).transform(train_y)
    train_X_, valid_X, train_y_, valid_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

    N_EPOCH = 1000
    BATCH_SIZE = 10
    N_BATCH = train_X_.shape[0] // BATCH_SIZE

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(N_EPOCH):
            train_X_, train_y_ = shuffle(train_X_, train_y_, random_state=456)
            for i in range(N_BATCH):
                ind_start = i * BATCH_SIZE
                ind_end = ind_start + BATCH_SIZE
                sess.run(train, feed_dict={X: train_X_[ind_start:ind_end], t: train_y_[ind_start:ind_end]})
            pred_y, valid_cost = sess.run([valid, cost], feed_dict={X: valid_X, t: valid_y})
            # pred_rate = sess.run(rate, feed_dict={p_y: lb().fit(ook).transform(pred_y), t: valid_y})
            # print("{epoch}:{rate}".format(epoch=epoch+1, rate=pred_rate))
            print("{epoch}:{cost}".format(epoch=epoch+1, cost=valid_cost))
        pred_y = sess.run(valid, feed_dict={X: test_X})

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
    print(float([t==p for t, p in zip(test_y_mini, pred_y)].count(True)) / float(len(test_y_mini)))
    print(f1_score(test_y_mini, pred_y, average='macro'))

def score_homework():
    train_X, test_X, train_y, test_y = load_mnist()
    pred_y = homework(train_X, train_y, test_X)
    print(f1_score(test_y, pred_y, average='macro'))


validate_homework()
# score_homework()