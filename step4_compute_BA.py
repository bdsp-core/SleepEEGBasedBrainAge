import datetime
import os
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.special import logsumexp
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('ticks')


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
        

def get_perc(BAI, age, sex, bmi, df_ref, ref_bai_col, plot=False, fig_path=None):
    if pd.notna(sex):
        df_ref = df_ref[df_ref.Sex==sex].reset_index(drop=True)
    else:
        df_ref = df_ref.copy()

    # similarity = exp(-(x-y)^2/sigma)

    # find sigma so that at bound, similarity=0.05
    sigma_age = -(10**2)/np.log(0.05)
    sigma_bmi = -(5**2)/np.log(0.05)

    cols = []
    log_sim = []
    if pd.notna(age):
        log_sim.append(-(df_ref.Age.values-age)**2/sigma_age)
    if pd.notna(bmi):
        log_sim.append(-(df_ref.BMI.values-bmi)**2/sigma_bmi)
    sim = np.exp(sum(log_sim)/len(log_sim))

    bai_ref = df_ref[ref_bai_col].values
    age_bound = np.abs(bai_ref).max()

    std = np.median(np.abs(bai_ref-np.median(bai_ref)))*1.4826
    kde = KernelDensity(bandwidth=std*0.4)
    kde.fit(bai_ref.reshape(-1,1), sample_weight=sim)
    bai_grid = np.linspace(-age_bound-10,age_bound+10,10000)
    bai_log_prob = kde.score_samples(bai_grid.reshape(-1,1))

    perc = np.exp(logsumexp(bai_log_prob[bai_grid<BAI]) - logsumexp(bai_log_prob))*100

    if plot:
        plt.close()
        fig = plt.figure(figsize=(10,8))

        ax = fig.add_subplot(111)
        res = ax.hist(bai_ref, bins=30, density=True, weights=sim)
        ax.plot(bai_grid, np.exp(bai_log_prob), c='k', lw=2)
        ax.plot([BAI]*2, [0,res[0].max()*1.01], ls='--', c='r', lw=2)
        ax.text(BAI, res[0].max()*1.01, f'{perc:.0f}%', ha='center', va='bottom', color='r', fontweight='bold')
        ax.set_xlabel('BAI (year)')
        ax.set_ylabel('Probability density')
        ax.set_xlim(-age_bound-1, age_bound+1)
        seaborn.despine()

        plt.tight_layout()
        if fig_path is None:
            plt.show()
        else:
            plt.savefig(fig_path)

    return perc, (bai_grid, bai_log_prob)

        
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

    df_ref = pd.read_csv('BAI_MGH_healthy.csv')
    
    df_res = df.drop(columns='SignalPath')
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
        age = df.Age.iloc[i]
        bmi = df.BMI.iloc[i]
        sex = df.Sex.iloc[i]
        if type(sex)==str:
            if sex.lower()=='m':
                sex = 1
            elif sex.lower()=='f':
                sex = 0
        if sex not in [0,1]:
            print(f'Unknown sex encoding (M or 1, F or 0): {sex}\nIgnore it.')
            sex = np.nan

        # adjust BA
        if pd.notna(age):
            idx = np.where((df_BA_adj.CA_min<=age)&(df_BA_adj.CA_max>age))[0][0]
            BA += df_BA_adj.bias.iloc[idx]

        # get BAI
        BAI = BA-age

        # get BAI percent
        BAI_perc = np.nan
        if pd.notna(BAI) and (pd.notna(age)|pd.notna(sex)|pd.notna(bmi)):
            BAI_perc, hist = get_perc(BAI, age, sex, bmi, df_ref, 'BAI_old', plot=True, fig_path=os.path.join(output_ba_dir, f'BAI_hist_{sid}.png'))

        df_res.loc[i, 'BA'] = BA
        df_res.loc[i, 'BAI'] = BAI
        df_res.loc[i, 'BAIPerc'] = BAI_perc
        df_res.loc[i, 'artifact_ratio'] = artifact_ratio
        df_res.loc[i, 'num_missing_stage'] = num_missing_stage

    print(df_res)
    now = datetime.datetime.now().strftime('%H%M%d_%m%d%Y')
    df_res.to_csv(os.path.join(output_ba_dir, f'BA_{now}.csv'), index=False)


if __name__=='__main__':
    main()

