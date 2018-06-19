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
    tf.test,
    tf.train
]

def homework(train_X, train_y, test_X):
    matmul = tf.matmul
    log_ = lambda x: tf.log(tf.clip_by_value(x, 1e-10, 1.))
    transpose = tf.transpose

    rng = np.random.RandomState(1234)

    class Autoencoder:
        # 初期化 パラメータＷは外から受け取る
        def __init__(self, vis_dim, hid_dim, W, function):
            super(Autoencoder, self).__init__()
            self.W = W
            self.a = tf.Variable(np.zeros(vis_dim).astype('float32'), name="a")
            self.b = tf.Variable(np.zeros(hid_dim).astype('float32'), name="b")
            self.function = function
            self.params = [self.W, self.a, self.b]

        def encode(self, x):
            u = matmul(x, self.W) + self.b
            return self.function(u)

        def decode(self, x):
            u = matmul(x, transpose(self.W)) + self.a
            return self.function(u)

        def f_prop(self, x):
            y = self.encode(x)
            return self.decode(y)

        def reconst_error(self, x, noise):
            tilde_x = x * noise
            reconst_x = self.f_prop(tilde_x)
            error = -tf.reduce_mean(tf.reduce_sum(x * log_(reconst_x) + (1. - x) * log_(1. - reconst_x), axis=1))
            return error, reconst_x

    class Dense:
        # 初期化 入力次元,出力次元,活性化関数
        def __init__(self, in_dim, out_dim, function):
            super(Dense, self).__init__()
            self.W = tf.Variable(rng.uniform(low=-.08, high=.08, size=(in_dim, out_dim)).astype('float32'))
            self.function = function

            self.model = Autoencoder(in_dim, out_dim, self.W, function)
            self.params = self.model.params

            self.b = self.model.b

        def f_prop(self, x):
            u = matmul(x, self.W) + self.b
            return self.function(u)

        def pretrain(self, x, noise):
            cost, reconst_x = self.model.reconst_error(x, noise)
            return cost, reconst_x

    # 確率勾配降下法
    # 更新したいパラメータのリスト,誤差関数,学習率
    def sdg(params, cost, eps):
        d_params = tf.gradients(cost, params)
        updates = []
        for d_param, param in zip(d_params, params):
            if d_param != None:
                updates.append(param.assign_add(-eps * d_param))
        return updates

    # 隠れ層のユニット数
    U_HIDDEN = 50
    layers = [
        Dense(784, U_HIDDEN, tf.nn.sigmoid),
        Dense(U_HIDDEN, U_HIDDEN, tf.nn.sigmoid),
        Dense(U_HIDDEN, 10, tf.nn.softmax)
    ]

    # 学習データ 教師なし
    X = np.copy(train_X)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # 事前学習
    for l, layer in enumerate(layers[:-1]):
        # 損傷率 データに加えるノイズの量
        CORRUPTION_LEVEL = np.float(0.3)
        BATCH_SIZE = 10
        N_BATCHES = X.shape[0] // BATCH_SIZE
        N_EPOCHS = 100
        EPS = 0.01

        x = tf.placeholder(tf.float32)
        noise = tf.placeholder(tf.float32)

        # 事前学習の定義 教師なし
        cost, reconst_x = layer.pretrain(x, noise)
        train = sdg(layer.params, cost, EPS)

        # 次の層へ入力するため現在の層を通して次元を変形
        encode = layer.f_prop(x)

        # 事前学習の実行
        print("START Pre-training[in layer{}]".format(l+1))
        for epoch in range(N_EPOCHS):
            X = shuffle(X, random_state=42)
            for i in range(N_BATCHES):
                start = i * BATCH_SIZE
                end = start + BATCH_SIZE

                _, error = sess.run([train, cost], feed_dict={x: X[start:end], noise: rng.binomial(size=X[start:end].shape, n=1, p=1.-CORRUPTION_LEVEL)})
            if epoch%10==0: print("Epoch{ep}: Error={er}".format(ep=epoch+1, er=error))
        print("FINISHI Pre-training[in layer{}]".format(l+1))

        # 次の層へ入力するため現在の層を通して次元を変形
        X = sess.run(encode, feed_dict={x: X})

    # 事後学習 教師あり
    BATCH_SIZE = 10
    N_BATCHES = train_X.shape[0] // BATCH_SIZE
    N_EPOCHS = 100
    EPS = 0.01

    x = tf.placeholder(tf.float32, [None, 784])
    t = tf.placeholder(tf.float32, [None, 10])

    def f_props(layers, x):
        params = []
        for layer in layers:
            x = layer.f_prop(x)
            params += layer.params
        return x, params

    y, params = f_props(layers, x)

    # 誤差関数の定義
    cost = -tf.reduce_mean(tf.reduce_sum(t * log_(y), axis=1))
    # パラメータの更新方法の定義
    updates = sdg(params, cost, EPS)

    # 学習
    train = tf.group(*updates)
    # 予測値 onehot表現から1桁数字に変形
    valid = tf.argmax(y, axis=1)

    print("START Fine-tuning")
    for epoch in range(N_EPOCHS):
        train_X, train_y = shuffle(train_X, train_y, random_state=63)
        for i in range(N_BATCHES):
            start = i * BATCH_SIZE
            end = start + BATCH_SIZE
            _, error = sess.run([train, cost], feed_dict={x: train_X[start:end], t: train_y[start:end]})
        pred_y = sess.run(valid, feed_dict={x: test_X})
        if epoch%10==0: print("Epoch{ep}: Error={er}".format(ep=epoch+1, er=error))
    print("FINISH Fine-tuning")

    return pred_y


def load_mnist():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    mnist_X = np.r_[mnist.train.images, mnist.test.images]
    mnist_y = np.r_[mnist.train.labels, mnist.test.labels]
    return train_test_split(mnist_X, mnist_y, test_size=0.2, random_state=42)

def validate_homework():
    train_X, test_X, train_y, test_y = load_mnist()

    # validate for small dataset
    train_X_mini = train_X[:100]
    train_y_mini = train_y[:100]
    test_X_mini = test_X[:100]
    test_y_mini = test_y[:100]

    pred_y = homework(train_X_mini, train_y_mini, test_X_mini)
    print(f1_score(np.argmax(test_y_mini, 1), pred_y, average='macro'))

def score_homework():
    train_X, test_X, train_y, test_y = load_mnist()
    pred_y = homework(train_X, train_y, test_X)
    print(f1_score(np.argmax(test_y, 1), pred_y, average='macro'))


validate_homework()