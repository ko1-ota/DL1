def homework(train_X, train_y, test_X):
    global num_words # =10000
    rng = np.random.RandomState(3456)
    from datetime import datetime
    t_start = datetime.now()


    class Embedding:
        def __init__(self, vocab_size, dim_emb, scale=0.08):
            self.V = tf.Variable(rng.randn(vocab_size, dim_emb).astype('float32') * scale, name="V")

        def f_prop(self, x):
            return tf.nn.embedding_lookup(self.V, x)

    def orthological_initializer(shape, scale=1.0):
        a = np.random.normal(0.0, 1.0, shape).astype(np.float32)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == shape else v
        return scale * q

    class RNN:
        def __init__(self, dim_in, dim_hid, m):
            self.dim_in = dim_in
            self.dim_hid = dim_hid
            self.W_h = tf.Variable(orthological_initializer([dim_hid, dim_hid]), name="W_h")
            self.W_x = tf.Variable(rng.uniform(low=-np.sqrt(6./(dim_in+dim_hid)), high=np.sqrt(6./(dim_in+dim_hid)), size=[dim_in, dim_hid]).astype('float32'), name="W_x")
            self.b = tf.Variable(np.zeros(dim_hid).astype('float32'), name="b")
            self.m = m

        def f_prop(self, x):
            def fn(h_tm1, x_and_m):
                x = x_and_m[0]
                m = x_and_m[1]
                h_t = tf.nn.tanh(tf.matmul(h_tm1, self.W_h) + tf.matmul(x, self.W_x) + self.b)
                return m[:, None] * h_t + (1 - m[:, None]) * h_tm1
            _x = tf.transpose(x, perm=[1,0,2])
            _m = tf.transpose(self.m)
            h_0 = tf.matmul(x[:, 0, :], tf.zeros([self.dim_in, self.dim_hid]))
            h = tf.scan(fn=fn, elems=[_x, _m], initializer=h_0)
            return h[-1]

    class Dense:
        def __init__(self, dim_in, dim_out, function=lambda x: x):
            self.W = tf.Variable(rng.uniform(low=-np.sqrt(6./(dim_in + dim_out)), high=np.sqrt(6./(dim_in + dim_out)), size=[dim_in, dim_out]).astype('float32'), name="W")
            self.b = tf.Variable(np.zeros(dim_out).astype('float32'), name="b")
            self.function = function

        def f_prop(self, x):
            u = tf.matmul(x, self.W) + self.b
            return self.function(u)

    DIM_EMB = 100
    DIM_HID = 50

    x = tf.placeholder(tf.int32, [None, None], name="x")
    t = tf.placeholder(tf.float32, [None, None], name="t")
    m = tf.cast(tf.not_equal(x, -1), tf.float32)

    layers = [
        Embedding(num_words, DIM_EMB),
        RNN(DIM_EMB, DIM_HID, m=m),
        Dense(DIM_HID, 1, tf.nn.sigmoid)
    ]

    def f_props(x):
        for layer in layers:
            x = layer.f_prop(x)
        return x

    y = f_props(x)

    _log = lambda x: tf.log(tf.clip_by_value(x, 1e-10, 1.0))
    cost = tf.reduce_mean(-t * _log(y) - (1. - t) * _log(1. - y))
    train = tf.train.AdamOptimizer().minimize(cost)
    test = tf.round(y)

    lens_train_X = [len(com) for com in train_X]
    sorted_train_index = sorted(range(len(lens_train_X)), key=lambda x: -lens_train_X[x])
    train_X = [train_X[ind] for ind in sorted_train_index]
    train_y = [train_y[ind] for ind in sorted_train_index]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    N_EPOCHS = 2000
    BATCH_SIZE = 5
    N_BATCHES_TRAIN = len(train_X) // BATCH_SIZE
    N_BATCHES_TEST = len(test_X) // BATCH_SIZE
    for epoch in range(N_EPOCHS):
        train_costs = []
        for i in range(N_BATCHES_TEST):
            start = i * BATCH_SIZE
            end = start + BATCH_SIZE

            train_X_mb = np.array(pad_sequences(train_X[start:end], padding='post', value=-1))
            train_y_mb = np.array(train_y[start:end])[:, np.newaxis]

            _, train_cost = sess.run([train, cost], feed_dict={x: train_X_mb, t: train_y_mb})
            train_costs.append(train_cost)
        print("EPOCH {epc}: cost={cst}".format(epc=epoch+1, cst=train_cost))

        pred_y = []
        for j in range(N_BATCHES_TEST):
            start = j * BATCH_SIZE
            end = start + BATCH_SIZE

            test_X_mb = np.array(pad_sequences(test_X[start:end], padding='post', value=-1))

            pred_y += sess.run(test, feed_dict={x: test_X_mb}).flatten().tolist()
        duration = datetime.now() -t_start
        if duration.seconds > 58*60:
            break

    # HINT: keras内の関数、pad_sequences は利用可能です。
    return pred_y
