# coding:utf-8
import numpy as np
        
def P():
    A = np.array([[7.], [7.]])
    B = np.array([[[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]]])
    C = np.array([4., 4., 4., 4.])
    D = np.array([5., 5., 5.])
    return C[:, np.newaxis] - D[np.newaxis, :]

print P()