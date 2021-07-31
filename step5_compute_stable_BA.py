import os
import pickle
import numpy as np
import pandas as pd
import sys
from load_dataset import *


if __name__ == '__main__':
    dataset = sys.argv[1].strip()
    
    output_ba_dir = 'output_BA'
    if not os.path.exists(output_ba_dir): os.mkdir(output_ba_dir)
    
    # load brain age model
    _get_brain_age_dir = eval(f'get_{dataset}_brain_age_dir')
    brain_age_dir = _get_brain_age_dir()
    feature_names = list(pd.read_csv(os.path.join(brain_age_dir, 'BA_features_used.csv')).feature.values)
    sys.path.insert(0, brain_age_dir)
    with open(os.path.join(brain_age_dir, 'stable_brain_age_model.pickle'), 'rb') as ff:
        model = pickle.load(ff)
    
    # get list of subjects
    df_mastersheet = pd.read_excel('mastersheet.xlsx')
    df_mastersheet = df_mastersheet[df_mastersheet.Dataset==dataset].reset_index(drop=True)
    
    # read features
    feature_dir = 'features'
    df_feat = pd.read_csv(os.path.join(feature_dir, f'combined_features_no_log_{dataset}.csv'))
    assert np.all(df_mastersheet.SID==df_feat.SID)
    
    # match the feature names in model
    df_feat = df_feat.rename(columns={x:x.replace('/','_') for x in df_feat.columns if '/' in x})
    
    stages = ['W','N1','N2','N3','R']
    _get_channels = eval(f'get_{dataset}_channels')
    ch_names, standard_ch_names, pair_ch_ids, combined_ch_names = _get_channels()
    for c1, c2 in zip(pair_ch_ids, combined_ch_names):
        for stage in stages:
            df_feat[f'kurtosis_{stage}_{c2}'] = (df_feat[f'kurtosis_{ch_names[c1[0]]}_{stage}']+df_feat[f'kurtosis_{ch_names[c1[1]]}_{stage}'])/2
        
    # read spindle features
    df_sp = pd.read_csv(os.path.join(feature_dir, f'spindle_features_N2_channel_avg_{dataset}.csv'))
    assert np.all(df_mastersheet.SID==df_sp.SID)
    
    # compute brain age
    # model contains all preprocessing and adjustment steps
    X = pd.concat([df_feat, df_sp], axis=1)[feature_names].values
    CA = df_mastersheet.Age.values
    BA = model.predict(X, y=CA)
    BAI = BA-CA
    
    df_feat['BA'] = BA
    df_feat['BAI'] = BAI
    df_feat = df_feat[['SID', 'Dataset', 'Age', 'Gender', 'BA', 'BAI', 'NumMissingStage', 'ArtifactRatio']]

    df_feat.to_csv(os.path.join(output_ba_dir, f'stable_BA_{dataset}.csv'), index=False)
    
