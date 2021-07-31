import os
import pickle
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
from load_dataset import *


#TODO sklearn.KNNImputer
def impute_missing_stage(X, K, Xnonan=None):
    nan_mask2D = np.isnan(X)
    nan_mask = np.any(nan_mask2D, axis=1)
    hasnan_ids = np.where(nan_mask)[0]
    nonan_ids = np.where(np.logical_not(nan_mask))[0]
    if Xnonan is None:
        Xnonan_ = X[nonan_ids]
    else:
        Xnonan_ = Xnonan
    X_ = np.array(X, copy=True)
    for i in hasnan_ids:
        row_nan_ids = np.isnan(X_[i])
        row_nonan_ids = np.logical_not(row_nan_ids)
        dists = np.linalg.norm(Xnonan_[:,row_nonan_ids]-X_[i,row_nonan_ids], axis=1, ord=2)
        X_[i,row_nan_ids] = Xnonan_[np.argsort(dists)[:K]].mean(axis=0)[row_nan_ids]
    if Xnonan is None:
        return X_, Xnonan_
    else:
        return X_
        
        
if __name__=='__main__':
    dataset = sys.argv[1].strip()
    
    output_ba_dir = 'output_BA'
    if not os.path.exists(output_ba_dir): os.mkdir(output_ba_dir)
    
    # load brain age model
    _get_brain_age_dir = eval(f'get_{dataset}_brain_age_dir')
    brain_age_dir = _get_brain_age_dir()
    Xnonan = sio.loadmat(os.path.join(brain_age_dir, 'mgh_eeg_data.mat'))['Xtr']
    KNN_K = 10  # number of patients without any missing stage to average
    with open(os.path.join(brain_age_dir, 'feature_normalizer_eeg.pickle'), 'rb') as f:
        feature_mean, feature_std = pickle.load(f)
    with open(os.path.join(brain_age_dir, 'brain_age_model.pickle'),'rb') as ff:
        brain_age_coef, brain_age_intercept = pickle.load(ff)
    df_BA_adj = pd.read_csv(os.path.join(brain_age_dir, 'BA_adjustment_bias.csv'))
    df_BA_adj.loc[0, 'CA_min'] = -np.inf
    df_BA_adj.loc[len(df_BA_adj)-1, 'CA_max'] = np.inf
        
    # get list of subjects
    df_mastersheet = pd.read_excel('mastersheet.xlsx')
    df_mastersheet = df_mastersheet[df_mastersheet.Dataset==dataset].reset_index(drop=True)
    
    # read features
    feature_dir = 'features'
    df_feat = pd.read_csv(os.path.join(feature_dir, f'combined_features_{dataset}.csv'))
    assert np.all(df_feat.SID==df_mastersheet.SID)
    X = df_feat.iloc[:,6:].values

    # pre-process
    if np.any(np.isnan(X)):
        #X = KNNImputer(n_neighbors=KNN_K).fit_transform(X)
        X = impute_missing_stage(X, KNN_K, Xnonan=Xnonan)
    X = (X-feature_mean)/feature_std

    # compute brain age
    BAs = np.logaddexp(np.dot(X, brain_age_coef)+brain_age_intercept, 0)

    # adjust BA
    BA_adjs = []
    for si in range(len(df_mastersheet)):
        CA = df_mastersheet.Age.iloc[si]
        idx = np.where((df_BA_adj.CA_min<=CA)&(df_BA_adj.CA_max>CA))[0][0]
        adj = df_BA_adj.bias.iloc[idx]
        BA_adjs.append(adj)

    df_feat['BA'] = BAs+np.array(BA_adjs)
    df_feat['BAI'] = df_feat.BA - df_mastersheet.Age

    df_feat = df_feat[['SID', 'Dataset', 'Age', 'Gender', 'BA', 'BAI', 'NumMissingStage', 'ArtifactRatio']]
    df_feat.to_csv(os.path.join(output_ba_dir, f'BA_{dataset}.csv'), index=False)

