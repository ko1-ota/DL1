import sys

from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
rng = np.random.RandomState(42)
random_state = 42

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

sys.modules['keras'] = None

def homework(train_X, train_y):
    global e_vocab_size, j_vocab_size, sess, x, d # 下記Checker Cell内 で定義した各変数を利用可能.

    from datetime import datetime as dt
    t_start = dt.now()
    rng = np.random.RandomState(6789)


    class Embedding:
        def __init__(self, vocab_size, dim_emb, scale=0.08):
            self.V = tf.Variable(rng.randn(vocab_size, dim_emb).astype('float32') * scale, name="V")
    
        def f_prop(self, x):
            return tf.nn.embedding_lookup(self.V, x)

    class LSTM:
        def __init__(self, dim_in, dim_hid, m, h_0=None, c_0=None):
            self.dim_in = dim_in
            self.dim_hid = dim_hid

            self.W_xi = tf.Variable(rng.uniform(low=-np.sqrt(6. / (dim_in+dim_hid)), high=np.sqrt(6. / (dim_in+dim_hid)), size=[dim_in, dim_hid]).astype('float32'), name="W_xi")
            self.W_hi = tf.Variable(rng.uniform(low=-np.sqrt(6. / (dim_in+dim_hid)), high=np.sqrt(6. / (dim_in+dim_hid)), size=[dim_in, dim_hid]).astype('float32'), name="W_hi")
            self.b_i = tf.Variable(np.zeros(dim_hid).astype('float32'), name="b_i")

            self.W_xf = tf.Variable(rng.uniform(low=-np.sqrt(6. / (dim_in+dim_hid)), high=np.sqrt(6. / (dim_in+dim_hid)), size=[dim_in, dim_hid]).astype('float32'), name="W_xf")
            self.W_hf = tf.Variable(rng.uniform(low=-np.sqrt(6. / (dim_in+dim_hid)), high=np.sqrt(6. / (dim_in+dim_hid)), size=[dim_in, dim_hid]).astype('float32'), name="W_hf")
            self.b_f = tf.Variable(np.zeros(dim_hid).astype('float32'), name="b_f")

            self.W_xo = tf.Variable(rng.uniform(low=-np.sqrt(6. / (dim_in+dim_hid)), high=np.sqrt(6. / (dim_in+dim_hid)), size=[dim_in, dim_hid]).astype('float32'), name="W_xo")
            self.W_ho = tf.Variable(rng.uniform(low=-np.sqrt(6. / (dim_in+dim_hid)), high=np.sqrt(6. / (dim_in+dim_hid)), size=[dim_in, dim_hid]).astype('float32'), name="W_ho")
            self.b_o = tf.Variable(np.zeros(dim_hid).astype('float32'), name="b_o")

            self.W_xc = tf.Variable(rng.uniform(low=-np.sqrt(6. / (dim_in+dim_hid)), high=np.sqrt(6. / (dim_in+dim_hid)), size=[dim_in, dim_hid]).astype('float32'), name="W_xc")
            self.W_hc = tf.Variable(rng.uniform(low=-np.sqrt(6. / (dim_in+dim_hid)), high=np.sqrt(6. / (dim_in+dim_hid)), size=[dim_in, dim_hid]).astype('float32'), name="W_hc")
            self.b_c = tf.Variable(np.zeros(dim_hid).astype('float32'), name="b_c")

            self.m = m
            self.h_0 = h_0
            self.c_0 = c_0

        def f_prop(self, x):
            def fn(h_and_c_tm1, x_and_m):
                h_tm1 = h_and_c_tm1[0]
                c_tm1 = h_and_c_tm1[1]
                x_t = x_and_m[0]
                m_t = x_and_m[1]

                i_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xi) + tf.matmul(h_tm1, self.W_hi) + self.b_i)
                f_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xf) + tf.matmul(h_tm1, self.W_hf) + self.b_f)
                o_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xo) + tf.matmul(h_tm1, self.W_ho) + self.b_o)

                c = f_t * c_tm1 + i_t * tf.nn.tanh(tf.matmul(x_t, self.W_xc) + tf.matmul(h_tm1, self.W_hc) + self.b_c)
                c = m_t[:, np.newaxis] * c + (1. - m_t[:, np.newaxis]) * c_tm1

                h = o_t * tf.nn.tanh(c)
                h = m_t[:, np.newaxis] * h + (1. - m_t[:, np.newaxis]) * h_tm1

                return [h, c]

            _x = tf.transpose(x, perm=[1,0,2])
            _m = tf.transpose(self.m)

            if self.h_0==None:
                self.h_0 = tf.matmul(x[:,0,:], tf.zeros([self.dim_in, self.dim_hid]))
            if self.c_0==None:
                self.c_0 = tf.matmul(x[:,0,:], tf.zeros([self.dim_in, self.dim_hid]))

            h_t, c_t = tf.scan(fn=fn, elems=[_x, _m], initializer=[self.h_0, self.c_0])
            return tf.transpose(h_t, perm=[1,0,2]), tf.transpose(c_t, perm=[1,0,2])

    class Dense:
        def __init__(self, dim_in, dim_out, function=lambda x: x):
            self.W = tf.Variable(rng.uniform(low=-np.sqrt(6. / (dim_in+dim_out)), high=np.sqrt(6. / (dim_in+dim_out)), size=[dim_in, dim_out]).astype('float32'), name="W")
            self.b = tf.Variable(np.zeros(dim_out).astype('float32'), name="b")
            self.function = function

        def f_prop(self, x):
            return self.function(tf.einsum('ijk,kl->ijl', x, self.W) + self.b)
        
    class Dropout:
        def __init__(self, keep_prob):
            self.keep_prob = keep_prob
            
        def f_prop(self, x):
            return tf.nn.dropout(x, keep_prob=self.keep_prob)

    m = tf.cast(tf.not_equal(x, -1), tf.float32)

    d_in = d[:, :-1]
    d_out = d[:, 1:]

    def f_props(layers, x):
        for layer in layers:
            x = layer.f_prop(x)
        return x

    dim_emb = 256
    dim_hid = 256

    encoder = [
        Embedding(e_vocab_size, dim_emb),
        Dropout(0.9),
        LSTM(dim_emb, dim_hid, m)
    ]

    h_enc, c_enc = f_props(encoder, x)

    decoder_1 = [
        Embedding(j_vocab_size, dim_emb),
        Dropout(0.9),
        LSTM(dim_emb, dim_hid, tf.ones_like(d_in, dtype='float32'), h_0=h_enc[:, -1, :], c_0=c_enc[:, -1, :])
    ]
    decoder_2 = [
        Dense(dim_hid, j_vocab_size, tf.nn.softmax),
        Dropout(0.8)
    ]

    h_dec, c_dec = f_props(decoder_1, d_in)
    y = f_props(decoder_2, h_dec)

    _log = lambda z: tf.log(tf.clip_by_value(z ,1e-10, 1.0))

    train_cost = -tf.reduce_mean(tf.reduce_sum(tf.one_hot(d_out, depth=j_vocab_size, dtype='float32') * _log(y), axis=[1,2]))

    train = tf.train.AdamOptimizer().minimize(train_cost)

    train_X_lens = [len(com) for com in train_X]
    sorted_train_indexes = sorted(range(len(train_X_lens)), key=lambda x: -train_X_lens[x])

    train_X = [train_X[ind] for ind in sorted_train_indexes]
    train_y = [train_y[ind] for ind in sorted_train_indexes]
       
    N_EPOCHS = 2000
    BATCH_SIZE = 5
    N_BATCHES = len(train_X) // BATCH_SIZE

    sess.run(tf.global_variables_initializer())
    for epoch in range(N_EPOCHS):
        train_costs = []
        for i in range(N_BATCHES):
            start = i * BATCH_SIZE
            end = start + BATCH_SIZE

            train_X_mb = np.array(pad_sequences(train_X[start:end], padding='post', value=-1))
            train_y_mb = np.array(pad_sequences(train_y[start:end], padding='post', value=-1))

            _, tc = sess.run([train, train_cost], feed_dict={x: train_X_mb, d: train_y_mb})

            train_costs.append(tc)
        duration = dt.now() - t_start
        if duration.seconds > 55*60:
            break
#         print('epoch: {epc}, cost={cst}'.format(epc=epoch+1, cst=np.mean(train_costs)))

    cost = train_cost
    return cost # 返り値のcostは,tensorflowの計算グラフのcostを返す.def homework(train_X, train_y):

def build_vocab(file_path):
    vocab = set()
    for line in open(file_path, encoding='utf-8'):
        words = line.strip().split()
        vocab.update(words)

    w2i = {w: np.int32(i+2) for i, w in enumerate(vocab)}
    w2i['<s>'], w2i['</s>'] = np.int32(0), np.int32(1)

    return w2i

def encode(sentence, w2i):
    encoded_sentence = []
    for w in sentence:
        encoded_sentence.append(w2i[w])
    return encoded_sentence

def load_data(file_path, vocab=None, w2i=None):
    if vocab is None and w2i is None:
        w2i = build_vocab(file_path)
    
    data = []
    for line in open(file_path, encoding='utf-8'):
        s = line.strip().split()
        s = ['<s>'] + s + ['</s>']
        enc = encode(s, w2i)
        data.append(enc)
    i2w = {i: w for w, i in w2i.items()}
    return data, w2i, i2w

def validate_homework():
    global e_vocab_size, j_vocab_size, sess, x, d
    
    train_X, e_w2i, e_i2w = load_data('train.en')
    train_y, j_w2i, j_i2w = load_data('train.ja')
    
    e_vocab_size = len(e_w2i)
    j_vocab_size = len(j_w2i)

    train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

    # validate for small dataset
    train_X_mini = train_X[:100]
    train_y_mini = train_y[:100]
    test_X_mini = test_X[:100]
    test_y_mini = test_y[:100]
    
    tf.reset_default_graph()

    with tf.Session() as sess:
        x = tf.placeholder(tf.int32, [None, None], name='x')
        d = tf.placeholder(tf.int32, [None, None], name='d')

        cost = homework(train_X_mini, train_y_mini)
        # sess, x, d はグローバル変数で定義されているので homework関数内で利用可能.
        # 返り値のcostは,tensorflowの計算グラフのcostを返す.
        
        test_batch_size = 128
        n_batches_test = -(-len(test_X_mini) // test_batch_size)
        test_costs = []
        
        for i in range(n_batches_test):
            start = i * test_batch_size
            end = start + test_batch_size if (start + test_batch_size) < len(test_X_mini) else len(test_X_mini)
            
            test_X_padded = np.array(pad_sequences(test_X_mini[start:end], padding='post', value=-1))
            test_y_padded = np.array(pad_sequences(test_y_mini[start:end], padding='post', value=-1))
            
            test_cost = sess.run(cost, feed_dict={x: test_X_padded, d: test_y_padded})
            test_costs.append(test_cost)
            
    print(np.mean(test_costs))

def score_homework():
    global e_vocab_size, j_vocab_size, sess, x, d
    
    train_X, e_w2i, e_i2w = load_data('/root/userspace/chap10/train.en')
    train_y, j_w2i, j_i2w = load_data('/root/userspace/chap10/train.ja')
    
    e_vocab_size = len(e_w2i)
    j_vocab_size = len(j_w2i)

    train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)
    
    train_X = train_X[:len(train_X)]
    train_y = train_y[:len(train_y)]
    test_X = test_X[:len(test_X)]
    test_y = test_y[:len(test_y)]
    
    tf.reset_default_graph()

    with tf.Session() as sess:
        x = tf.placeholder(tf.int32, [None, None], name='x')
        d = tf.placeholder(tf.int32, [None, None], name='d')

        cost = homework(train_X, train_y)
        # sess, x, d はグローバル変数で定義されているので homework関数内で利用可能.
        # 返り値のcostは,tensorflowの計算グラフのcostを返す.
        
        test_batch_size = 128
        n_batches_test = -(-len(test_X) // test_batch_size)
        test_costs = []
        
        for i in range(n_batches_test):
            start = i * test_batch_size
            end = start + test_batch_size if (start + test_batch_size) < len(test_X) else len(test_X)
            
            test_X_padded = np.array(pad_sequences(test_X[start:end], padding='post', value=-1))
            test_y_padded = np.array(pad_sequences(test_y[start:end], padding='post', value=-1))
            
            test_cost = sess.run(cost, feed_dict={x: test_X_padded, d: test_y_padded})
            test_costs.append(test_cost)
            
    print(np.mean(test_costs))

# score_homework()
validate_homework()