import numpy as np
from numpy import linalg as la
import math

def normalize(x):
    res = []
    for row in x:
        res.append(np.divide(row,np.sqrt(np.sum(row**2))))
    return res
#OK
def discretizationEigenVectorData(eigenVector):
    Y = np.array([0.0 for _ in range(len(eigenVector[0])) for _ in range(len(eigenVector))])
    Y = np.reshape(Y, (len(eigenVector), len(eigenVector[0])))
    j = []
    for row in eigenVector:
        j.append(np.unravel_index(row.argmax(), row.shape)[0])
    for it, i in enumerate(j):
        Y[it][i] = 1
    return Y 
#OK
def discretization(eigenVectors):
              
    eigenVectors = normalize(np.array(eigenVectors, float))
    n = len(eigenVectors)
    k = len(eigenVectors[0])
    R = np.array([0.0 for _ in range(k) for _ in range(k)])
    R = np.reshape(R,  (k, k))
    R[:,0] = np.transpose(eigenVectors[math.floor(float(n)/2)])
    c = np.array([0 for _ in range(n)])
    c = np.reshape(c, (n,1))
    for j in range(1,k):
        c = c + abs(np.matmul(eigenVectors, np.reshape(R[:,j-1], (k,1))))    
        i = np.unravel_index(c.argmin(), c.shape)
        R[:,j] = np.transpose(eigenVectors[i[0]])
    lastObjectiveValue = 0.0
    for i in range(20):
        eigenDiscrete = discretizationEigenVectorData(np.matmul(eigenVectors, R))
        u,s,v = la.svd(np.matmul(np.transpose(eigenDiscrete), eigenVectors))
        v = np.transpose(v)
        NcutValue = 2*(n-np.sum(s))
        if abs(NcutValue-lastObjectiveValue) < np.finfo(float).eps:
            break
        
        lastObjectiveValue = NcutValue
        R = np.matmul(v, np.transpose(u))
    return eigenDiscrete
    
def SpectralClustering(affinity, K, type=3):
    
    d = np.sum(affinity, axis=1)
    d[d == 0] = np.finfo(float).eps
            
    D = np.diag(d)
    L = D - affinity
    if type == 1:
        NL = L
    elif type == 2:
        Di = np.diag(1/d)
        NL = Di * L
    elif type == 3:
        Di = np.diag(1/np.sqrt(d))
        NL = np.matmul(Di,L)
        NL = np.matmul(NL,Di)
    #giù
    eigval, eigvec = la.eig(NL)
    eigval = eigval.real
    eigvec = eigvec.real
    eigabs = abs(eigval)
    res = sorted(range(len(eigabs)), key=lambda k: eigabs[k])
    U = eigvec[:,res[0:K]]
    if type == 3:
        U = normalize(U)
    U = np.array(U)
    eigDiscrete = discretization(U)
    #su
    labels = []
    for row in eigDiscrete:
        labels.append(np.unravel_index(row.argmax(), row.shape)[0])
    for it, i in enumerate(labels):
        labels[it] = i + 1
    return labels