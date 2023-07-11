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

# this is an example but in reality, generate this programmably
df = pd.DataFrame(data={
    'SID':[os.path.basename(x[:x.rfind('.')]) for x in signal_paths],
    'Age':[np.nan]*N,
    'SignalPath':signal_paths,
    })

df.to_excel('mastersheet.xlsx', index=False)
