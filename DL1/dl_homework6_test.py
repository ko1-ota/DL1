from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data

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
    rng = np.random.RandomState(1234)
    _train_X, valid_X, _train_y, valid_y = train_test_split(train_X, train_y, test_size=0.1, random_state=467)

    class Conv:
        def __init__(self, filter_shape, function=tf.nn.relu, strides=[1,1,1,1], padding='VALID'):
            len_in = np.prod(filter_shape[:3])
            len_out = np.prod(filter_shape[:2])*filter_shape[3]

            self.W = tf.Variable(rng.uniform(size=filter_shape, low=-.08, high=.08).astype('float32'), name="W")
            self.b = tf.Variable(np.zeros(filter_shape[3]).astype('float32'), name="b")
            self.function = function
            self.strides = strides
            self.padding = padding

        def prop(self, x):
            u = tf.nn.conv2d(x, self.W, strides=self.strides, padding=self.padding)
            return self.function(u)

    class Pool:
        def __init__(self, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID'):
            self.ksize = ksize
            self.strides = strides
            self.padding = padding

        def prop(self, x):
            return tf.nn.avg_pool(x, ksize=self.ksize, strides=self.strides, padding=self.padding)

    class Flatten:
        def prop(self, x):
            return tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))

    class Dense:
        def __init__(self, dim_in, dim_out, function=tf.nn.softmax):
            self.W = tf.Variable(rng.uniform(size=[dim_in, dim_out], low=-.08, high=.08).astype('float32'), name="W")
            self.b = tf.Variable(np.zeros(dim_out).astype('float32'), name="b")
            self.function = function

        def prop(self, x):
            u = tf.matmul(x, self.W) + self.b
            return self.function(u)

    class Dropout:
        def __init__(self, keep_prob):
            self.keep_prob = keep_prob

        def prop(self, x):
            return tf.nn.dropout(x, self.keep_prob)


    layers = [
        Conv([5,5,1,10]), # 28x28x1 -> 24x24x10
        Pool(), # ->12x12x10
        Conv([5,5,10,20]), # -> 8x8x20
        Pool(), # 4x4x20
        Flatten(), # 4*4*20=320
        Dropout(0.7),
        Dense(4*4*20,10) # -> 10
    ]

    x = tf.placeholder(tf.float32, [None,28,28,1])
    t = tf.placeholder(tf.float32, [None,10])

    def props(layers, x):
        for layer in layers:
            x = layer.prop(x)
        return x

    y = props(layers, x)

    _log = lambda x: tf.log(tf.clip_by_value(x, 1e-10, 1.0))
    cost = - tf.reduce_mean(tf.reduce_sum(t * _log(y), axis=1))
    EPS = 0.01
    train = tf.train.GradientDescentOptimizer(EPS).minimize(cost)

    valid = tf.argmax(y, axis=1)

    N_EPOCHS = 2000
    BATCH_SIZE = 100
    N_BATCHES = train_X.shape[0] // BATCH_SIZE

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print("START TRAINING")
    for epoch in range(N_EPOCHS):
        train_X, train_y = shuffle(_train_X, _train_y, random_state=55)
        for i in range(N_BATCHES):
            start = i * BATCH_SIZE
            end = start + BATCH_SIZE
            sess.run(train, feed_dict={x: train_X[start:end], t: train_y[start:end]})
        pred_y = sess.run(valid, feed_dict={x: test_X})
        error = sess.run(cost, feed_dict={x: valid_X, t: valid_y})
        print("TRAINING>>Epoch:{epc}, Error:{err}".format(epc=epoch+1, err=error))
    print("FINISH TRAINING")

    return pred_y

def load_mnist():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    mnist_X = np.r_[mnist.train.images, mnist.test.images]
    mnist_y = np.r_[mnist.train.labels, mnist.test.labels]
    return train_test_split(mnist_X, mnist_y, test_size=0.2, random_state=42)

def validate_homework():
    train_X, test_X, train_y, test_y = load_mnist()
    train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
    test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))

    # validate for small dataset
    train_X_mini = train_X[:1000]
    train_y_mini = train_y[:1000]
    test_X_mini = test_X[:1000]
    test_y_mini = test_y[:1000]

    pred_y = homework(train_X_mini, train_y_mini, test_X_mini)
    print(f1_score(np.argmax(test_y_mini, 1), pred_y, average='macro'))

def score_homework():
    train_X, test_X, train_y, test_y = load_mnist()
    train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
    test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))
    
    pred_y = homework(train_X, train_y, test_X)
    print(f1_score(np.argmax(test_y, 1), pred_y, average='macro'))

validate_homework()
# score_homework()