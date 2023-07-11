import datetime
import os
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm


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
        
        
def main():
    output_ba_dir = 'output_BA'
    os.makedirs(output_ba_dir, exist_ok=True)
    
    # load brain age model
    brain_age_dir = 'brain_age_model_fco'
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
    df = pd.read_excel('mastersheet.xlsx')
    feature_dir = 'features'
    
    df_res = df[['SID', 'Age']]
    for i in tqdm(range(len(df))):
        sid = df.SID.iloc[i]
        # read features
        df_feat = pd.read_csv(os.path.join(feature_dir, f'features_{sid}.csv'))
        X = df_feat.iloc[:,1:].values
        mat = sio.loadmat(os.path.join(feature_dir, f'features_{sid}.mat'), variable_names=['artifact_ratio', 'num_missing_stage'])
        artifact_ratio = float(mat['artifact_ratio'])
        num_missing_stage = int(mat['num_missing_stage'])

        # pre-process
        if np.any(np.isnan(X)):
            #X = KNNImputer(n_neighbors=KNN_K).fit_transform(X)
            X = impute_missing_stage(X, KNN_K, Xnonan=Xnonan)
        X = (X-feature_mean)/feature_std

        # compute brain age
        BA = np.logaddexp(np.dot(X, brain_age_coef)+brain_age_intercept, 0)[0]

        # adjust BA
        if 'Age' in df.columns:
            CA = df.Age.iloc[i]
            if pd.notna(CA):
                idx = np.where((df_BA_adj.CA_min<=CA)&(df_BA_adj.CA_max>CA))[0][0]
                BA += df_BA_adj.bias.iloc[idx]

        df_res.loc[i, 'BA'] = BA
        df_res.loc[i, 'artifact_ratio'] = artifact_ratio
        df_res.loc[i, 'num_missing_stage'] = num_missing_stage

    print(df_res)
    now = datetime.datetime.now().strftime('%H%M%d_%m%d%Y')
    df_res.to_csv(os.path.join(output_ba_dir, f'BA_{now}.csv'), index=False)


if __name__=='__main__':
    main()

