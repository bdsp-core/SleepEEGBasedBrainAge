"""
generate an excel file with columns:
SID, Dataset, Age, Gender, SignalPath, AnnotPath
"""
import os
import glob
import numpy as np
import pandas as pd


cwd = os.getcwd()
signal_paths = glob.glob(os.path.join(cwd, '*.edf'))
N = len(signal_paths)

# create symbolic link for files with space
for i in range(N):
    fn = os.path.basename(signal_paths[i])
    folder = os.path.dirname(signal_paths[i])
    if ' ' in fn:
        new_path = os.path.join(folder, fn.replace(' ', ''))
        os.symlink(signal_paths[i], new_path)
        signal_paths[i] = new_path

# this is an example but in reality, generate this programmably
df = pd.DataFrame(data={
    'SID':[os.path.basename(x[:x.rfind('.')]) for x in signal_paths],
    'Age':[np.nan]*N,            # required
    'Sex':[np.nan]*N,            # optional, can be empty
    'BMI':[np.nan]*N,            # optional, can be empty
    'SignalPath':signal_paths,   # must be absolute path
    # if no staging file, do automatic sleep staging
    #'AnnotPath':annot_paths,    # must be absolute path  # different datasets have different format
    })

# output
df.to_excel('mastersheet.xlsx', index=False)
