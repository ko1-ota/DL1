# coding:utf-8
import numpy as np
        
def P():
    A = np.array([[0, 3, 0], [5, 5, 2]])
    B = np.array([[2, 0, 1], [5, 4, 3]])
    C = np.array([[9, 9], [9, 9]])
    D = np.array([[7, 7], [7, 7]])
    return len(np.where(A!=0)[0])

print P()