# %%
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

#%%
parser = argparse.ArgumentParser()

parser.add_argument('--datafiles', type=str, nargs='+', help='Training datafile')
parser.add_argument('--labelsfile', type=str, help='Sample labels')

parser.add_argument(
    '--cv_k', type=np.int, default=5, help='Number of CV folds (default: %(default)s)'
)
parser.add_argument(
    '--cv_n', type=np.int, default=10, help='Number of CV cycles (default: %(default)s)'
)

args = parser.parse_args()
DATAFILES = args.datafiles
LABELSFILE = args.labelsfile

CV_K = args.cv_k
CV_N = args.cv_n
#%%

for datafile in DATAFILES:

    original_data = pd.read_csv(datafile, sep='\t', header=0, index_col=0)
    original_labels = pd.read_csv(LABELSFILE, header=None)

    ys = []
    for i in range(CV_N):
        ys.append(original_labels)

    #%%
    for n in tqdm(range(CV_N)):
        skf = StratifiedKFold(CV_K, shuffle=True, random_state=n)

        for i, (idx_tr, idx_ts) in enumerate(skf.split(original_data, ys[n])):

            cv_data = original_data.iloc[idx_tr]
            cv_labels = original_labels.iloc[idx_tr]

            layer_filename = os.path.splitext(datafile)[0]
            layer_ext = os.path.splitext(datafile)[1]
            layer_filename_new = f'{layer_filename}_{n}_{i}{layer_ext}'

            labels_filename = os.path.splitext(LABELSFILE)[0]
            labels_ext = os.path.splitext(LABELSFILE)[1]
            labels_filename_new = f'{labels_filename}_{n}_{i}{labels_ext}'

            cv_data.to_csv(layer_filename_new, sep='\t')
            cv_labels.to_csv(labels_filename_new, sep='\t', header=None, index=False)
# %%
