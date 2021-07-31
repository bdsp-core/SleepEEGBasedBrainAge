"""
generate an excel file with columns:
SID, Dataset, Age, Gender, SignalPath, AnnotPath
"""
import numpy as np
import pandas as pd


# this is an example but in reality, generate this programmably
df = pd.DataFrame(
    data=[
    ['SSC_4704_1', 'ApoE', 50, 'M', '/data/brain_age/brain_age_AllMGH/general_BA_code/example_data/ApoE/edfs/SSC_4704_1.EDF', '/data/brain_age/brain_age_AllMGH/general_BA_code/example_data/ApoE/labels/SSC_4704_1.STA'],
    ['STNF00104_1', 'STAGES', 50, 'F', '/data/brain_age/brain_age_AllMGH/general_BA_code/example_data/STAGES/STNF/STNF00104_1.edf', '/data/brain_age/brain_age_AllMGH/general_BA_code/example_data/STAGES/STNF/STNF00104_1.csv'],
    ['A0001_4_165907', 'WSC', 50, 'M', '/data/brain_age/brain_age_AllMGH/general_BA_code/example_data/WSC/edfs/A0001_4 165907.EDF', '/data/brain_age/brain_age_AllMGH/general_BA_code/example_data/WSC/labels/A0001_4 165907.STA']
    ],
    columns=['SID', 'Dataset', 'Age', 'Gender', 'SignalPath', 'AnnotPath']
    )

df.to_excel('mastersheet.xlsx', index=False)
