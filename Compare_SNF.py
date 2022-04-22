# This takes the results of SNF.py and snf_integration.r and compares them

import snf
from data_convert import load_data_txt, load_mat
import argparse
import numpy as np
import os

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
affinity_networks = snf.make_affinity(datas.data, K=20, mu=0.5)
fused_network = snf.snf(affinity_networks, K=20)
m = []
for row in fused_network:
    m.append(np.amax(row))
W_scale = fused_network/max(m)
        
paths = []
for f in range(3):
    paths.append(OUT_DIR + "INF_" + args.LAYERS + "_tr_" + str(CV_N) + "_" + str(CV_K) + "_" + str(f) + "_mat.txt")
paths.append(OUT_DIR + "INF_" + args.LAYERS + "_tr_" + str(CV_N) + "_" + str(CV_K) + ".txt")
paths.append(OUT_DIR + "INF_" + args.LAYERS + "_tr_" + str(CV_N) + "_" + str(CV_K) + "_similarity_mat_fused.txt")
        
mats = load_mat(paths)
        
outf = open(OUT_DIR + "Diff" + str(CV_N) + "_" + str(CV_K) + ".txt", "w")
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
     
outf.close()
        
    