# coding: utf-8


import numpy as np


def q1_1a():
    return np.arange(10)//2

def q1_1b():
    return (np.arange(5)[:, np.newaxis] * (1, 1)).flatten()

def q1_1c():
    return np.array([(i, i) for i in range(5)]).flatten()

def q1_2a():
    x = np.arange(10)
    return x * (x % 2 == 1)

def q1_2b():
    return np.array([i if i % 2 == 0 else 0 for i in range(10)])

def q1_2c():
    return np.arange(10) * np.tile([0, 1], 5)

def q1_3a():
    return 2 ** (np.arange(5))

def q1_3b():
    return np.apply_along_axis(np.square, 0, (np.arange(5)))

def q1_3c():
    x = [1]
    [x.append(x[i] * 2) for i in range(4)]
    return np.array(x)

def q1_4a():
    return np.sign(np.arange(10) - 4.5)

def q1_4b():
    return np.concatenate([-np.ones(5), np.ones(5)])

def q1_4c():
    return (np.arange(10) > 4.5) * 2.0 -1


def q2(A, B, C, x):
    return np.einsum('ik,k,kj->ij', A, x, B) - C.T


def q3(x, a):
    distlist = []
    for i in range(len(x)):
        dist = np.linalg.norm(x[i] - a)
        distlist.append(dist)
    y = x[np.argsort(np.array(distlist))]
    return y


def Q4(M, N, x0, x1, y0, y1):
    A = np.arange(M*N).reshape(N, M)
    B = A[y0:y1+1, x0:x1+1]
    return np.sum(B)


def Q5(x):
    A = np.zeros((len(x), max(x)+1))
    B = np.tile(np.arange(3), (9, 1))
    M = np.array(B==np.array(x)[:,np.newaxis])
    return A + M * 1

def Q5_inv(y):
    return np.array(np.where(y==1)[1])


def Q6(m, n):
    xm = np.arange(m) + 1
    xn = np.arange(n) + 1
    return xm[np.newaxis, :] * xn[:, np.newaxis]


def Q7(x):
    return (np.append(x, 0) - np.append(0, x))[1:len(x)]


if __name__ == '__main__':
    x = np.array([1,1,2,2,3,4,6,8,9])
    print Q7(x)


    # print Q6(4, 5)


    # x = np.array([0, 1, 2, 1, 0, 2, 2, 1, 0])
    # y = np.array([[ 1., 0., 0.], [ 0.,  1.,  0.], [ 0.,  0.,  1.], [ 0.,  1.,  0.], [ 1.,  0.,  0.], [ 0.,  0.,  1.], [ 0.,  0.,  1.], [ 0.,  1.,  0.], [ 1.,  0.,  0.]])
    # print Q5(x)
    # print Q5_inv(y)

    # print Q4(8, 6, 1, 3, 1, 2)


    # x = np.array([[87, 14], [81, 62], [81, 18], [ 8, 63], [51, 15], [38, 63], [80, 36], [69, 78], [26, 9]])
    # a = np.array([25, 75])

    # print q3(x, a)


    # A = np.array([[1,2,3],[4,5,6]])
    # B = np.array([[1,2],[3,4],[5,6]])
    # C = np.array([[2,3],[1,2]])
    # x = np.array([-1,1,3])

    # print q2(A, B, C, x)


    # print q1_1a()
    # print q1_1b()
    # print q1_1c()

    # print q1_2a()
    # print q1_2b()
    # print q1_2c()

    # print q1_3a()
    # print q1_3b()
    # print q1_3c()

    # print q1_4a()
    # print q1_4b()
    # print q1_4c()