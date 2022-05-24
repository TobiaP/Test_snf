from matplotlib.pyplot import axis
import numpy as np


def estimateNumberOfClustersGivenGraph(W, NUMC={2,3,4,5}):
    
    W = (W+np.transpose(W))/2
    for i in range(len(W)):
        W[i][i] = 0
    
    if(len(NUMC)<0):
        degs = sum(W, axis = 1)
        
    degs[degs==0] = np.finfo(float).eps
    D = np.diag(degs)
    
    