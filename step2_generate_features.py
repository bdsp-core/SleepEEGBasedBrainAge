import os
from collections import Counter
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('ticks')
from load_dataset import *
from segment_EEG import segment_EEG
from extract_features_parallel import extract_features


def myprint(epoch_status):
    sm = Counter(epoch_status)
    for k, v in sm.items():
        print(f'{k}: {v}/{len(epoch_status)}, {v*100./len(epoch_status)}%')


if __name__=='__main__':
    epoch_length = 30 # [s]
    amplitude_thres = 500 # [uV]
    line_freq = 60.  # [Hz]
    bandpass_freq = [0.5, 20.]  # [Hz]
    n_jobs = 8
    normal_only = True
    newFs = 200.

    # get list of files
    df_mastersheet = pd.read_excel('mastersheet.xlsx')
            
    # define output folder
    output_feature_dir = 'features'
    if not os.path.exists(output_feature_dir): os.mkdir(output_feature_dir)
    output_sleep_vis_dir = 'sleep_spectrogram_hypnogram'
    if not os.path.exists(output_sleep_vis_dir): os.mkdir(output_sleep_vis_dir)

    # for each recording
    for si in tqdm(range(len(df_mastersheet))):
        sid = df_mastersheet.SID.iloc[si]
        dataset = df_mastersheet.Dataset.iloc[si]
        signal_path = df_mastersheet.SignalPath.iloc[si]
        annot_path = df_mastersheet.AnnotPath.iloc[si]
        feature_path = os.path.join(output_feature_dir, 'features_'+sid+'.mat')
        figure_path = os.path.join(output_sleep_vis_dir, sid+'.png')
        
        if os.path.exists(feature_path):
            mat = sio.loadmat(feature_path, variable_names=['sleep_stages', 'epoch_status'])
            sleep_stages = mat['sleep_stages'].flatten()
            epoch_status = mat['epoch_status'].flatten()
            
        else:
            # load dataset
            _load_dataset = eval(f'load_{dataset}_dataset')
            EEG, sleep_stages, EEG_channels, combined_EEG_channels, Fs, start_time = _load_dataset(signal_path, annot_path)

            # segment EEG
            epochs, sleep_stages, epoch_start_idx, epoch_status, specs, freq = segment_EEG(EEG, sleep_stages, epoch_length, epoch_length, Fs, newFs, notch_freq=line_freq, bandpass_freq=bandpass_freq, amplitude_thres=amplitude_thres, n_jobs=n_jobs)
            if epochs.shape[0] <= 0:
                raise ValueError('Empty EEG segments')
            Fs = newFs

            # plot spectrogram and sleep stages
            plt.close()
            fig = plt.figure(figsize=(10,6))
            gs = fig.add_gridspec(2, 1, height_ratios=[1,2])
            tt = np.arange(len(sleep_stages))*epoch_length/3600
            
            ax_ss = fig.add_subplot(gs[0])
            ax_ss.step(tt, sleep_stages, color='k')
            ax_ss.yaxis.grid(True)
            ax_ss.set_ylim([0.5,5.5])
            ax_ss.set_yticks([1,2,3,4,5])
            ax_ss.set_yticklabels(['N3', 'N2', 'N1', 'R', 'W'])
            ax_ss.set_xlim([tt.min(), tt.max()])
            plt.setp(ax_ss.get_xlabel(), visible=False)
            
            specs_db = 10*np.log10(specs)
            specs_db = specs_db.mean(axis=-1)  # average across channels
            ax_spec = fig.add_subplot(gs[1], sharex=ax_ss)
            ax_spec.imshow(
                    specs_db.T, aspect='auto', origin='lower', cmap='jet',
                    vmin=-5, vmax=25,
                    extent=(tt.min(), tt.max(), freq.min(), freq.max()))
            ax_spec.set_ylabel('freq (Hz)')
            ax_spec.set_xlabel('time (hour)')
            
            plt.tight_layout()
            plt.savefig(figure_path, bbox_inches='tight', pad_inches=0.05)

            if normal_only:
                good_ids = np.where(epoch_status=='normal')[0]
                if len(good_ids)<=300:
                    myprint(epoch_status)
                    raise ValueError('<=300 normal epochs')
                epochs = epochs[good_ids]
                specs = specs[good_ids]
                sleep_stages = sleep_stages[good_ids]
                epoch_start_idx = epoch_start_idx[good_ids]

            # extract brain age features
            features, feature_names = extract_features(
                epochs, Fs, EEG_channels, 2,
                2, 1, return_feature_names=True,
                combined_channel_names=combined_EEG_channels,
                n_jobs=n_jobs, verbose=True)
            
            myprint(epoch_status)
            sio.savemat(feature_path, {
                'start_time':start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'EEG_feature_names':feature_names,
                'EEG_features':features,
                'EEG_specs':specs,
                'EEG_frequency':freq,
                'sleep_stages':sleep_stages,
                'epoch_start_idx':epoch_start_idx,
                'age':df_mastersheet.Age.iloc[si],
                'gender':df_mastersheet.Gender.iloc[si],
                'epoch_status':epoch_status,
                'Fs':Fs,
                })
                
                
    # get stage-averaged features
    
    stages = ['W','N1','N2','N3','R']
    stage2num = {'W':5,'R':4,'N1':3,'N2':2,'N3':1}
    num2stage = {stage2num[x]:x for x in stage2num}
    minimum_epochs_per_stage = 5
    
    datasets = df_mastersheet.Dataset.unique()
    for dataset in datasets:
        print(f'converting into stage-averaged features for {dataset}')
        
        df = df_mastersheet[df_mastersheet.Dataset==dataset].reset_index(drop=True)
        
        ba_features = []
        ba_features_no_log = []
        artifact_ratios = []
        num_missing_stages = []
        for si in tqdm(range(len(df))):
            sid = df.SID.iloc[si]
            feature_path = os.path.join(output_feature_dir, 'features_'+sid+'.mat')
            
            # if for any reason the feature file does not exist,
            # fill this row with nan
            if not os.path.exists(feature_path):
                D = len(ba_features[-1])  #TODO assume the previous feature file is found
                ba_features.append([np.nan]*D)
                ba_features_no_log.append([np.nan]*D)
                artifact_ratios.append(np.nan)
                num_missing_stages.append(np.nan)
                continue
                
            mat = sio.loadmat(feature_path)
            features = mat['EEG_features']
            sleep_stages = mat['sleep_stages'].flatten()
            epoch_status = np.char.strip(mat['epoch_status'])
            feature_names = np.char.strip(mat['EEG_feature_names'])
            
            artifact_ratio = 1-len(sleep_stages)/len(epoch_status)
            num_missing_stage = 5-len(set(sleep_stages[~np.isnan(sleep_stages)]))
                
            # log-transform brain age features
            features_no_log = np.array(features)
            features = np.sign(features)*np.log1p(np.abs(features))
            
            # average features across sleep stages
            X = []
            X_no_log = []
            for stage in stages:
                ids = sleep_stages==stage2num[stage]
                if ids.sum()>=minimum_epochs_per_stage:
                    X.append(np.nanmean(features[ids], axis=0))
                    X_no_log.append(np.nanmean(features_no_log[ids], axis=0))
                else:
                    X.append(np.zeros(features.shape[1])+np.nan)
                    X_no_log.append(np.zeros(features.shape[1])+np.nan)
            X = np.concatenate(X)
            X_no_log = np.concatenate(X_no_log)
            
            ba_features.append(X)
            ba_features_no_log.append(X_no_log)
            artifact_ratios.append(artifact_ratio)
            num_missing_stages.append(num_missing_stage)
        df['ArtifactRatio'] = artifact_ratios
        df['NumMissingStage'] = num_missing_stages
        
        cols = np.concatenate([[x.strip()+'_'+stage for x in feature_names] for stage in stages])
        df_feat = pd.DataFrame(data=np.array(ba_features), columns=cols)
        df_feat = pd.concat([df[['SID', 'Dataset', 'Age', 'Gender', 'ArtifactRatio', 'NumMissingStage']], df_feat], axis=1)
        df_feat.to_csv(os.path.join(output_feature_dir, f'combined_features_{dataset}.csv'), index=False)
        
        df_feat.loc[:,cols] = np.array(ba_features_no_log)
        df_feat.to_csv(os.path.join(output_feature_dir, f'combined_features_no_log_{dataset}.csv'), index=False)
        
