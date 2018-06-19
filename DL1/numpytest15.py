# coding:utf-8
import numpy as np
        
def P():
    A = np.array([[7.], [7.]])
    B = np.array([[[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]]])
    C = np.array([[9, 9], [9, 9]])
    D = np.array([[7, 7], [7, 7]])
    return A[:, np.newaxis] - B

print P()