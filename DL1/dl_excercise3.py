from sklearn.utils import shuffle

import numpy as np

np.random.seed(1234)


# XOR
train_X = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
train_y = np.array([[1], [1], [0], [0]])

def softmax(x):
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x, axis=1, keepdims=True)

test_X, test_y = train_X, train_y

# Layer1 weights
W1 = np.random.uniform(low=-0.08, high=0.08, size=(2, 3)).astype('float32')
b1 = np.zeros(3).astype('float32')

# Layer2 weights
W2 = np.random.uniform(low=-0.08, high=0.08, size=(3, 1)).astype('float32')
b2 = np.zeros(1).astype('float32')


def train(x, t, eps=1.0):
    global W1, b1, W2, b2 # to access variables that defined outside of this function.
    
    # Forward Propagation Layer1
    u1 = np.matmul(x, W1) + b1
    z1 = sigmoid(u1)
    
    # Forward Propagation Layer2
    u2 = np.matmul(z1, W2) + b2
    z2 = sigmoid(u2)
    
    # Back Propagation (Cost Function: Negative Loglikelihood)
    y = z2
    cost = np.sum(-t*np.log(y) - (1 - t)*np.log(1 - y))
    delta_2 = y - t # Layer2 delta
    delta_1 = deriv_sigmoid(u1) * np.matmul(delta_2, W2.T) # Layer1 delta

    # Update Parameters Layer1
    dW1 = np.matmul(x.T, delta_1)
    db1 = np.matmul(np.ones(len(x)), delta_1)
    W1 = W1 - eps*dW1
    b1 = b1 - eps*db1
    
    # Update Parameters Layer2
    dW2 = np.matmul(z1.T, delta_2)
    db2 = np.matmul(np.ones(len(z1)), delta_2)
    W2 = W2 - eps*dW2
    b2 = b2 - eps*db2

    return cost

def test(x, t):
    # Forward Propagation Layer1
    u1 = np.matmul(x, W1) + b1
    z1 = sigmoid(u1)
    
    # Forward Propagation Layer2
    u2 = np.matmul(z1, W2) + b2
    z2 = sigmoid(u2)
    
    y = z2
    
    # Test Cost
    cost = np.sum(-t*np.log(y)-(1-t)*np.log(1-y))
    return cost, y

# Epoch
for epoch in range(2000):
    # Online Learning
    for x, y in zip(train_X, train_y):
        cost = train(x[np.newaxis, :], y[np.newaxis, :])
    cost, pred_y = test(test_X, test_y)

print(pred_y)