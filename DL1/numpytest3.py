# coding:utf-8
import numpy as np
        
def P():
    A = np.array([[8, 5, 3], [1, 9, 0], [4, 2, 7], [10, 6, 11]])
    return np.sort(A.flatten()).reshape((4, 3))

print P()