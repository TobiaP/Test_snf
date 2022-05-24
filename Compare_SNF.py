# This takes the results of SNF.py and snf_integration.r and compares them

import snf
from snf import get_n_clusters
from calNMI import calNMI
from sklearn.cluster import spectral_clustering
from sklearn.metrics import v_measure_score, normalized_mutual_info_score
from data_convert import load_data_txt, load_mat
import argparse
import numpy as np
import os
from SpectralClustering import SpectralClustering
from snf_tuning import snf_tuning
from timeit import default_timer as timer
class myArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(myArgumentParser, self).__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line):
        for arg in line.split():
            if not arg.strip():
                continue
            if arg[0] == '#':
                break
            yield arg
            
parser = myArgumentParser(
    description='compares results from SNF.py and snf_integration.r .',
    fromfile_prefix_chars='@',
)
parser.add_argument('--DATAFILE', type=str, help='Training datafile')
parser.add_argument('--OUTDIR', type=str, help='Output directory')
parser.add_argument('--DATASET', type=str, help='dataset')
parser.add_argument('--TARGET', type=str, help='target')
parser.add_argument('--SPLIT_ID', type=str, help='split_id')
parser.add_argument('--LAYERS', type=str, help='layers')
parser.add_argument('--MODEL', type=str, help='model')
parser.add_argument('--OUTFILE', type=str, help='outfile')
parser.add_argument('--LAB', type=str, help="lab file")
parser.add_argument('--CLUST', type=str, help="Clusteringmethod on fused graph")
parser.add_argument('--CLUSTINFO', type=str, help="Should the number of clusters be equal to the number of classes?")
parser.add_argument('--THREADS', type=str, help="Number of threads for rSNF")

args = parser.parse_args()
CV_K = args.OUTFILE[-5]
CV_N = args.OUTFILE[-7]
LAYERS = args.LAYERS.split("_")
DATASET = args.DATASET
TARGET = args.TARGET
SPLIT_ID = args.SPLIT_ID
MODEL = args.MODEL
IN_DIR = args.DATAFILE + "/" + DATASET + "/" + TARGET + "/" + SPLIT_ID + "/"
OUT_DIR = args.OUTDIR + "/" + DATASET + "/" + TARGET + "/" + MODEL + "/" + SPLIT_ID + "/rSNF/"
LAB = args.LAB
clustMethod = args.CLUST
clustInfo = args.CLUSTINFO
threads = args.THREADS
clustinfo = False

def difference(mat1, mat2):
    avg_diff = 0
    avg_per = 0
    for row in range(len(mat1)):
        div = abs(mat2[row]-mat1[row])
        avg_diff += np.mean(div)
        per = (100 * div)/mat1[row]
        avg_per += np.mean(per)
    avg_diff /= len(mat1)
    avg_per /= len(mat1)
    outstr = "Mean difference: " + str(avg_diff) + "\naverage percentage difference: " + str(avg_per) + "\n"
    print(outstr)
    return outstr

files = []
for l in range(3): 
    files.append(IN_DIR  + LAYERS[l] + "_tr_" + str(CV_N) + "_" + str(CV_K) + ".txt")
datas = load_data_txt(files[0], files[1], files[2], LAB)

# tuning
print("Start Tuning\n")
opt_par = snf_tuning(datas.data, datas.labels, clustMethod, clustInfo)

K = opt_par.best_K
mu = opt_par.best_alpha
print("End Tuning K = " + str(K) + " alpha = " + str(mu) + "\n")

# create similarity (affinity) networks
print("### Similarity graphs creation")
start0 = timer()
affinity_networks = snf.make_affinity(datas.data, metric='euclidean', K=K, mu=mu)
end = timer()

# network fusion
print("### Graph fusion")
start = timer()
fused_network = snf.snf(affinity_networks, K=K)
end = timer()
print(f"{end - start:.3f}s elapsed")

# rescaled matrix
fused_network_sc = fused_network / np.max(fused_network)

# clustering 
if clustMethod == 'spectral':
    if clustInfo:
        nclust = len(np.unique(datas.labels))
        lab = SpectralClustering(fused_network, nclust)
    else:
        print("### Estimating number of clusters")
        start = timer()
        best, _ = get_n_clusters(fused_network)
        print("### Spectral clustering")
        fused_labels = spectral_clustering(fused_network, n_clusters=best, assign_labels='discretize', random_state=11)
        end0 = timer()
        print(f"{end0 - start:.3f}s elapsed")
        nmi = v_measure_score(datas.labels, fused_labels)  
        print()
        print(f"{end0 - start0:.3f}s elapsed overall")
        print()
        print(f"Est. number of clusters = {best}")
        print(f"NMI = {nmi:.5f}")

# read R files
p = open(OUT_DIR + "INF_" + args.LAYERS + "_tr_" + str(CV_N) + "_" + str(CV_K) + "_NMI_score.txt")
labsR = p.readlines()
p.close()
Rlabs = []
for it, label in enumerate(lab):
    Rlabs.append(int(labsR[it+1]))

# output        
paths = []
for f in range(3):
    paths.append(OUT_DIR + "INF_" + args.LAYERS + "_tr_" + str(CV_N) + "_" + str(CV_K) + "_" + str(f) + "_mat.txt")
paths.append(OUT_DIR + "INF_" + args.LAYERS + "_tr_" + str(CV_N) + "_" + str(CV_K) + ".txt")
paths.append(OUT_DIR + "INF_" + args.LAYERS + "_tr_" + str(CV_N) + "_" + str(CV_K) + "_similarity_mat_fused.txt")
        
mats = load_mat(paths)
        
p = open(OUT_DIR + "INF_" + args.LAYERS + "_tr_" + str(CV_N) + "_" + str(CV_K) + "_NMI_score.txt")
mats.append(float(p.readline()))
p.close()        
        
p = open(OUT_DIR + "INF_" + args.LAYERS + "_tr_" + str(CV_N) + "_" + str(CV_K) + "_OPT.txt")
mats.append(p.readlines())
p.close()        
        
outf = open(OUT_DIR + "Diff" + str(CV_N) + "_" + str(CV_K) + ".txt", "w")
outf.write("opt_parameters difference\n")
print("R parameters: " + str(mats[6]) + "\n" + "Python parameters: K: " + str(K) + " alpha : " + str(mu) + "\n")
outf.write("Differences between affinity matrices:\n")
print("Differences between affinity matrices:\n")
outf.write(difference(mats[0], affinity_networks[0]))
outf.write(difference(mats[1], affinity_networks[1]))
outf.write(difference(mats[2], affinity_networks[2]))
outf.write("Differences between snf:\n")
print("Differences between snf:\n")
outf.write(difference(mats[3], fused_network))
outf.write("Differences between scaled snf:\n")
print("Differences between scaled snf:\n")
outf.write(difference(mats[4], W_scale))
print("Cluster Differences\n" + str(calNMI(lab, Rlabs)) + "\n")
outf.write("Cluster Differences\n" + str(calNMI(lab, Rlabs)) + "\n")
print("SNFNMI_allfeats:\n")
outf.write("SNFNMI_allfeats:\n" + str(mats[5] - SNFNMI_allfeats))
print(str(mats[5]) + " " + str(SNFNMI_allfeats) + " " + str(mats[5] - SNFNMI_allfeats))
     
outf.close()
        
    