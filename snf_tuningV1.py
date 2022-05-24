
import snf
from snf import get_n_clusters
from snf.metrics import nmi
from data_convert import load_data_txt
from random import seed, randint
import numpy as np
from sklearn.metrics import v_measure_score, normalized_mutual_info_score
from joblib import Parallel, delayed
from sklearn.utils import Bunch
from sklearn.cluster import spectral_clustering
from calNMI import calNMI 
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def snf_cv(W, lab, clm, infocl, K=5, N=10):
    median_NMI = []
    #nclust = get_n_clusters(W)
    #group_k = SpectralClustering(W, nclust[0])
    #median_NMI = calNMI(group_k, lab)
    
    for n in range(N):
        skf = StratifiedKFold(K, shuffle=True, random_state=n)
        SNFNMI_K = []
        for i, (train_index, test_index) in enumerate(skf.split(W, lab)):
            W_k = W[np.ix_(test_index,test_index)]
            lab_k = lab[test_index]
            #if clm == 'spectral':
            #    if infocl:
            #        nclust = len(np.unique(lab))
            #        group_k = SpectralClustering(W_k, nclust)
            #    else:
            best, _ = get_n_clusters(W_k)
            #OK
            group_k = spectral_clustering(W_k, n_clusters=best, assign_labels='discretize', random_state=i)
            SNFNMI_K.append(v_measure_score(lab_k, group_k))
        median_NMI.append(np.median(np.array(SNFNMI_K)))
    result = np.median(np.array(median_NMI))
    print(result)
    
    return result
            

def NMI_tuning(K, alpha, distL, lab, clm, infocl):
    affinityL = snf.make_affinity(distL, metric='euclidean', K=K, mu=alpha)
    W_K = snf.snf(affinityL, K=K)
    #OK
    print(str(K) + " " + str(alpha))
    val = snf_cv(W_K, lab, clm=clm, infocl=infocl)
    return val

def snf_tuning(distL, lab, clm, infocl):
    #min and max K values
    minK = 10
    maxK = 30
    stepK = 1
    K_values = np.arange(minK, maxK+stepK, stepK)
    
    #min and max alpha values
    min_alpha = 0.3
    max_alpha = 0.8
    step_alpha = 0.05
    alpha_values = np.arange(min_alpha, max_alpha+step_alpha, step_alpha)
    alpha_values = alpha_values.round(2)
    #OK
    NMI_tun = Parallel(n_jobs=4)(delayed(NMI_tuning)(k, alpha, distL, lab, clm, infocl) for k in K_values for alpha in alpha_values)
    
    NMI_tun = np.array(NMI_tun)
    NMI_tun = NMI_tun.reshape((len(K_values), len(alpha_values)))
    nk = len(K_values)
    
    max_nmi_fk = []
    tab_median_NMI = []
    
    for elk in range(nk):
        max_nmi_fk.append(np.max(NMI_tun[elk]))
        tab_median_NMI.append(NMI_tun[elk])
    
    max_nmi_fk = np.array(max_nmi_fk)
    
    best_K_idx = np.unravel_index(max_nmi_fk.argmax(), max_nmi_fk.shape)
    best_K = K_values[best_K_idx]
    
    #for row in NMI_tun:
    best_alpha_idx = np.unravel_index(NMI_tun[best_K_idx].argmax(), NMI_tun[best_K_idx].shape)
    best_alpha = alpha_values[best_alpha_idx]
    print(NMI_tun)
    return Bunch(best_K=best_K, best_alpha=best_alpha, tab_median_NMI=tab_median_NMI)