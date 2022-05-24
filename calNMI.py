from cmath import nan
import numpy as np
from scipy import stats as st

def entropy(x):
    classx = np.unique(x)
    nx = len(x)
    nc = len(classx)
    
    prob = [nan for i in range(nc)]
    for i in range(nc):
        prob[i] = np.sum(x == classx[i])/nx
    
    result = -np.nansum(prob*np.log2(prob))
    return result
    

def mutualInformation(x, y):
    np.seterr(divide='ignore', invalid='ignore')
    classx = np.unique(x)
    classy = np.unique(y)
    ncx = len(classx)
    ncy = len(classy)
    nx = len(x)
    
    probxy = [ nan for i in range(ncx) for j in range(ncy)]
    probxy = np.array(probxy, float)
    probxy = np.array_split(probxy, ncx)
    
    for i in range(ncx):
        for j in range(ncy):
            probxy[i][j]=np.sum((x == classx[i]) & (y == classy[j]))/nx
    probx = np.nansum(probxy, axis=1)
    probx = np.transpose([probx]*ncy)
    proby = np.nansum(probxy, axis=0)
    proby = np.array([proby]*ncx)
    result = np.nansum(probxy*np.log2(np.divide(probxy,(probx*proby))))
    return result

def calNMI(x,y):
    x = np.array(x)
    y = np.array(y)
    result = np.max(np.array([0.,mutualInformation(x, y)/np.sqrt(entropy(x)*entropy(y))]))
    return result
    