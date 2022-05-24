import numpy as np
import snf
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import SpectralClustering
#from SpectralClustering import SpectralClustering
from sklearn.metrics import v_measure_score
from timeit import default_timer as timer
from sklearn.utils import Bunch
from joblib import Parallel, delayed

def snf_tuning_cv(databunch, SNF_K, SNF_mu, CV_N=10, CV_K=5):
    print(SNF_K + SNF_mu)
    SNF_K = np.int(SNF_K) # sanity check
    
    affinity = snf.make_affinity(databunch.data, metric='euclidean', K=SNF_K, mu=SNF_mu)
    W = snf.snf(affinity, K=SNF_K)
   
    median_nmi = list()
    for n in range(CV_N):
        skf = StratifiedKFold(CV_K, shuffle=True, random_state=n)

        snfnmi_k = list()
        for i, (train_index, test_index) in enumerate(skf.split(W, databunch.labels)):
            W_k = W[np.ix_(test_index, test_index)]
            lab_k = databunch.labels[test_index]
            best, _ = snf.get_n_clusters(W_k)
            
            fused_labels = SpectralClustering(n_clusters=best, eigen_solver = 'amg', assign_labels='discretize', random_state=i).fit_predict(W_k)
            nmi_k = v_measure_score(lab_k, fused_labels)
            snfnmi_k.append(nmi_k)

        median_nmi.append(np.median(np.array(snfnmi_k)))
    
    out_arr = np.median(np.array(median_nmi))
    return out_arr

def snf_tuning(dataset, CV_N=10, CV_K=5):
    K_min = 10
    K_max = 30
    K_step = 1

    mu_min = 0.3
    mu_max = 0.8
    mu_step = 0.05

    K_grid = np.arange(K_min, K_max + K_step, K_step)
    mu_grid = np.arange(mu_min, mu_max + mu_step, mu_step)

    results = Parallel(n_jobs=4)(delayed(snf_tuning_cv)(dataset, SNF_K, SNF_mu) for SNF_K in K_grid for SNF_mu in mu_grid)
    
    results = np.reshape(results, (len(K_grid), len(mu_grid)))
    
    max_nmi = 0
    for i, row in enumerate(results):
        for j, nmi in enumerate(row):
            if nmi > max_nmi:
                K_opt = K_grid[i]
                mu_opt = mu_grid[j]
                max_nmi = nmi

    return Bunch(K_opt = K_opt, mu_opt=mu_opt)