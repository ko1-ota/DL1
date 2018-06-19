# coding:utf-8
import numpy as np
        
def P():
    A = np.array([[8, 8], [8, 8]])
    B = np.array([[1, 1], [1, 1]])
    C = np.array([[9, 9], [9, 9]])
    D = np.array([[7, 7], [7, 7]])
    return C - np.dot(np.exp(A), B)

print P()