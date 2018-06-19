# coding:utf-8
import numpy as np
        
def P():
    A = np.array([[1, 3], [2, 0]])
    B = np.array([[2, 3], [1, 0]])
    C = np.array([[9, 9], [9, 9]])
    D = np.array([[7, 7], [7, 7]])
    return np.dot(np.linalg.inv(A), B)

print P()