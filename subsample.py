# This takes in input the name of some file paths, takes a maximum of 500 variables from it
# then delets the file and creates another one with the same name.
# The input files must be the 3 layer files so that every files has the same number of layers
import os
import random

def subsample(paths, cv_n, cv_k):
    
    for path_in in paths: 
        file = open(path_in)      # choice of random list of layers, max 500
        a = file.readline()
        b = a.split()
        file.close()
        num_layers = len(b)
        layers = random.sample(range(1, num_layers), min(num_layers-1, 500))
        layers.append(0)
        layers.sort()
        
        for i in range(cv_n):
            for j in range(cv_k):
                
                path = path_in[:-7]+(str(i)+"_"+str(j)+".txt")
                
                file = open(path)
                res_file = []
                lines = file.readlines()
                for line in lines:
                    res_line = []
                    l = line.split()
                    for lay in layers:
                        res_line.append(l[lay])
                    res_file.append(res_line)
                    
                file.close()
                os.remove(path)
                file = open(path, "w")
                for line in res_file:
                    file.write(" ".join(line)+"\n")
                file.close()
        
    