
# After the prepare_data_snf.py generates thhhe .txt files this 
# uses those files to create a 3-dimensional array to be used for
# the SNF.py library

import os
from re import L
from sklearn.utils import Bunch
import numpy as np

def load_data_txt(path_layer1, path_layer2, path_layer3, path_lab):
    
    for pat in [path_layer1, path_layer2, path_layer3, path_lab]:
        if not os.path.isfile(pat):
            raise ValueError('{} is not a valid dataset.'.format(pat))
        
    layer1 = open(path_layer1)
    layer2 = open(path_layer2)
    layer3 = open(path_layer3)
    lab = open(path_lab)
    
    data = list()   
    
    for iteration, file in enumerate([layer1, layer2, layer3]):
        
        lines = file.readlines()
        data_lay = list()
        for iteration, line in enumerate(lines):
            if not iteration == 0:
                data_feat = line.split()
                for i, d in enumerate(data_feat) :
                    if not i == 0:
                        data_feat[i]=float(d) 
                data_lay.append(data_feat[1:])
        data.append(np.array(data_lay))
        
    labels = lab.read().split()
    for i,  d in enumerate(labels) : 
        labels[i] = int(d)
    
    return Bunch(data=data, labels=np.array(labels))

def load_mat(paths):

    files = []
    
    for path in paths:
        files.append(open(path))
    
    mats = []
    for file in files:
        lines = file.readlines()
        rows = []
        for row in range(1, len(lines)):
            line = lines[row].split("\t")
            l = [float(x) for x in line[1:]]
            l = np.array(l)
            rows.append(l)
        mats.append(np.array(rows))
    return mats