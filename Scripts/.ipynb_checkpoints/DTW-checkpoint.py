import time
import numpy as np

def DTW(output, target, window):
    n, m = len(output), len(target)
    w = np.max([window, abs(n-m)])
    dtw_matrix = np.zeros((n+1, m+1))
    dtw_matrix += float("Inf")
    dtw_matrix[0, 0] = 0
    for i in range(1, n+1):
        a, b = np.max([1, i-w]), np.min([m, i+w])+1
        dtw_matrix[i,a:b] = 0
        
        
        for j in range(a, b):
            cost = np.abs(output[i-1] - target[j-1])
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
            
    return dtw_matrix[-1, -1]