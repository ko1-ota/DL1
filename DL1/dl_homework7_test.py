import sys

from keras.datasets import cifar10
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf

del [
    tf.app,
    tf.compat,
    tf.contrib,
    tf.errors,
    tf.gfile,
    tf.graph_util,
    tf.image,
    tf.layers,
    tf.logging,
    tf.losses,
    tf.metrics,
    tf.python_io,
    tf.resource_loader,
    tf.saved_model,
    tf.sdca,
    tf.sets,
    tf.summary,
    tf.sysconfig,
    tf.test
]


def homework(train_X, train_y, test_X):
    from datetime import datetime
    t_start = datetime.now()
    rs = np.random.RandomState(2222)


    class ZCAWhitening:
        def __init__(self, eps=1e-4):
            self.eps = eps
            self.mean = None
            self.ZCA_matrix = None

        def fit(self, x):
            x = x.reshape(x.shape[0], -1)
            self.mean = np.mean(x, axis=0)
            x -= self.mean
            A, d, _ = np.linalg.svd(np.dot(x.T, x)/x.shape[0])
            self.ZCA_matrix = np.dot(np.dot(A, np.diag(1. / np.sqrt(d + self.eps))), A.T)

        def transform(self, x):
            x_shape = x.shape
            x = x.reshape(x_shape[0], -1)
            x -= self.mean
            return np.dot(x, self.ZCA_matrix).reshape(x_shape)

        def fit_transform(self, x):
            self.fit(x)
            return self.transform(x)


    class Conv:
        def __init__(self, filter_shape, padding='VALID', strides=[1,1,1,1]):
            fan_in = np.prod(filter_shape[:3])
            fan_out = np.prod(filter_shape[:2])*filter_shape[3]
            self.W = tf.Variable(rs.uniform(low=-np.sqrt(6./(fan_in+fan_out)), high=np.sqrt(6./(fan_in+fan_out)), size=filter_shape).astype('float32'), name="W")
            self.b = tf.Variable(np.zeros(filter_shape[3]).astype('float32'), name="b")
            self.padding = padding
            self.strides = strides

        def f_prop(self, x):
            return tf.nn.conv2d(x, self.W, padding=self.padding, strides=self.strides) + self.b

    class Pooling:
        def __init__(self, ksize=[1,2,2,1], padding='VALID', strides=[1,2,2,1]):
            self.ksize = ksize
            self.padding = padding
            self.strides = strides

        def f_prop(self,x):
            return tf.nn.avg_pool(x, ksize=self.ksize, padding=self.padding, strides=self.strides)

    class BNorm:
        def __init__(self, shape, eps=np.float32(1e-5)):
            self.gamma = tf.Variable(np.ones(shape).astype('float32'), name="gamma")
            self.beta = tf.Variable(np.zeros(shape).astype('float32'), name="beta")
            self.eps = eps

        def f_prop(self, x):
            mean, var = tf.nn.moments(x, [0,1,2], keep_dims=True)
            return self.gamma * ((x - mean) / tf.sqrt(var + self.eps)) + self.beta

    class Flatten:
        def f_prop(self, x):
            return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

    class Dense:
        def __init__(self, dim_in, dim_out):
            self.W = tf.Variable(rs.uniform(low=-np.sqrt(6/(dim_in+dim_out)), high=np.sqrt(6/(dim_in+dim_out)), size=(dim_in, dim_out)).astype('float32'), name="W")
            self.b = tf.Variable(np.zeros(dim_out).astype('float32'), name="b")

        def f_prop(self, x):
            return tf.matmul(x, self.W) + self.b

    class Dropout:
        def __init__(self, keep_prob=0.5):
            self.keep_prob = keep_prob

        def f_prop(self, x):
            return tf.nn.dropout(x, keep_prob=self.keep_prob)

    class Activation:
        def __init__(self, function=lambda x: x):
            self.function = function

        def f_prop(self, x):
            return self.function(x)


    # _train_X, valid_X, _train_y, valid_y = train_test_split(train_X, train_y, test_size=0.2, random_state=3456)

    zcaw = ZCAWhitening()
    # zca_train_X = zcaw.fit_transform(_train_X)
    zca_train_X = zcaw.fit_transform(train_X)
    # zca_train_y = _train_y
    zca_train_y = train_y
    # zca_valid_X = zcaw.fit_transform(valid_X)
    # zca_valid_y = valid_y
    zca_test_X = zcaw.fit_transform(test_X)


    layers = [
        # input dimension: 32x32x3
        Conv([3,3,3,32]), # 30x30x32
        Activation(function=tf.nn.relu),
        Pooling(), # 15x15x32
        BNorm([15,15,32]),
        Conv([3,3,32,128]), # 13x13x128
        Activation(function=tf.nn.relu),
        Pooling(), # 6x6x128
        BNorm([6,6,128]),
        Conv([3,3,128,256]), # 4x4x256
        Activation(function=tf.nn.relu),
        Conv([1,1,256,256]), # 4x4x256
        Activation(function=tf.nn.relu),
        Conv([1,1,256,128]), # 4x4x128
        Activation(function=tf.nn.relu),
        Pooling(), # 2x2x128
        Flatten(), # 1x1x512
        Dense(512, 256), # 1x1x256
        Activation(function=tf.nn.relu),
        Dropout(keep_prob=0.8),
        Dense(256, 10), # 1x1x10
        Activation(function=tf.nn.softmax)
    ]

    x = tf.placeholder(tf.float32, [None,32,32,3])
    t = tf.placeholder(tf.float32, [None,10])

    def f_props(layers, x):
        for layer in layers:
            x = layer.f_prop(x)
        return x

    y = f_props(layers, x)

    _log = lambda x: tf.log(tf.clip_by_value(x, 1e-10, 1.0))
    cost = -tf.reduce_mean(tf.reduce_sum(t * _log(y), axis=1))
    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    valid = tf.argmax(y, axis=1)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    N_EPOCHS = 2000
    BATCH_SIZE = 100
    N_BATCHES = train_X.shape[0] // BATCH_SIZE
    for epoch in range(N_EPOCHS):
        # print("TRAINING epoch:{}".format(epoch+1))
        # train_X, train_y = shuffle(train_X, train_y, random_state=1212)
        zca_train_X, zca_train_y = shuffle(zca_train_X, zca_train_y, random_state=1212)
        for i in range(N_BATCHES):
            start = i * BATCH_SIZE
            end = start + BATCH_SIZE
            sess.run(train, feed_dict={x: zca_train_X[start:end], t: zca_train_y[start:end]})
        # pred_y, error = sess.run([valid, cost], feed_dict={x: zca_valid_X, t: zca_valid_y})
        pred_y = sess.run(valid, feed_dict={x: test_X})
        # print("FINISH   epoch:{ep} with error[{err}]".format(ep=epoch+1, err=error))
        # timer: finish in 60 min
        duration = datetime.now() - t_start
        if duration.seconds > 58*60:
            # print("duration is over")
            break
    pred_y = sess.run(valid, feed_dict={x: zca_test_X})

    return pred_y


sys.modules['keras'] = None

def load_cifar():
    (cifar_X_1, cifar_y_1), (cifar_X_2, cifar_y_2) = cifar10.load_data()

    cifar_X = np.r_[cifar_X_1, cifar_X_2]
    cifar_y = np.r_[cifar_y_1, cifar_y_2]

    cifar_X = cifar_X.astype('float32') / 255
    cifar_y = np.eye(10)[cifar_y.astype('int32').flatten()]

    train_X, test_X, train_y, test_y = train_test_split(cifar_X, cifar_y,
                                                        test_size=10000,
                                                        random_state=42)

    return (train_X, test_X, train_y, test_y)

def validate_homework():
    train_X, test_X, train_y, test_y = load_cifar()

    # validate for small dataset
    train_X_mini = train_X[:1000]
    train_y_mini = train_y[:1000]
    test_X_mini = test_X[:1000]
    test_y_mini = test_y[:1000]

    pred_y = homework(train_X_mini, train_y_mini, test_X_mini)
    print(f1_score(np.argmax(test_y_mini, 1), pred_y, average='macro'))

def score_homework():
    train_X, test_X, train_y, test_y = load_cifar()
    pred_y = homework(train_X, train_y, test_X)
    print(f1_score(np.argmax(test_y, 1), pred_y, average='macro'))


validate_homework()
# score_homework()