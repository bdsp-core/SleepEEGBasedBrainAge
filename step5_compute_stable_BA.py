import datetime
import os
import sys
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm


def main():
    output_ba_dir = 'output_BA'
    os.makedirs(output_ba_dir, exist_ok=True)
    
    # load brain age model
    brain_age_dir = 'brain_age_model_fco'
    feature_names = list(pd.read_csv(os.path.join(brain_age_dir, 'BA_features_used.csv')).feature.values)
    sys.path.insert(0, brain_age_dir)
    with open(os.path.join(brain_age_dir, 'stable_brain_age_model.pickle'), 'rb') as ff:
        model = pickle.load(ff)
    
    # get list of subjects
    df = pd.read_excel('mastersheet.xlsx')
    feature_dir = 'features'
    
    # read features
    df_res = df[['SID', 'Age']]
    for i in tqdm(range(len(df))):
        sid = df.SID.iloc[i]
        df_feat = pd.read_csv(os.path.join(feature_dir, f'features_{sid}_no_log.csv'))
        mat = sio.loadmat(os.path.join(feature_dir, f'features_{sid}.mat'),
                variable_names=['EEG_channels',
                    'combined_EEG_channels', 'combined_EEG_channels_ids',
                    'artifact_ratio', 'num_missing_stage'])
        combined_EEG_channels = np.char.strip(mat['combined_EEG_channels'].flatten())
        combined_EEG_channels_ids = mat['combined_EEG_channels_ids']
        EEG_channels = np.char.strip(mat['EEG_channels'].flatten())
        artifact_ratio = float(mat['artifact_ratio'])
        num_missing_stage = int(mat['num_missing_stage'])
        
        # match the feature names in model
        df_feat = df_feat.rename(columns={x:x.replace('/','_') for x in df_feat.columns if '/' in x})
        
        stages = ['W','N1','N2','N3','R']
        for c1, c2 in zip(combined_EEG_channels_ids, combined_EEG_channels):
            for stage in stages:
                df_feat[f'kurtosis_{stage}_{c2}'] = (df_feat[f'kurtosis_{EEG_channels[c1[0]]}_{stage}']+df_feat[f'kurtosis_{EEG_channels[c1[1]]}_{stage}'])/2
            
        # read spindle features
        df_sp = pd.read_csv(os.path.join(feature_dir, f'spindle_features_{sid}.csv'))
        
        # compute brain age
        # model contains all preprocessing and adjustment steps
        X = df_feat.merge(df_sp, on='SID', how='inner')[feature_names].values

        # COUPL_OVERLAP has different mean, remove that feature
        #for chn in combined_EEG_channels:
        #    if 'COUPL_OVERLAP_'+chn in feature_names:
        #        idx = feature_names.index('COUPL_OVERLAP_'+chn)
        #        X[:,idx] = model.steps[0][1].mean_[idx]

        CA = df.Age.iloc[i]
        if pd.isna(CA):
            CA = 70
        BA = model.predict(X, y=CA)[0]
        
        df_res.loc[i, 'RobustBA'] = BA
        df_res.loc[i, 'artifact_ratio'] = artifact_ratio
        df_res.loc[i, 'num_missing_stage'] = num_missing_stage

    print(df_res)
    now = datetime.datetime.now().strftime('%H%M%d_%m%d%Y')
    df_res.to_csv(os.path.join(output_ba_dir, f'robust_BA_{now}.csv'), index=False)
    

if __name__ == '__main__':
    main()

