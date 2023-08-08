import os
import sys
from collections import Counter
from itertools import groupby
import datetime
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
import joblib
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('ticks')
from myfunc import load_shhs_data
from segment_EEG import segment_EEG
from extract_features_parallel import extract_features
#sys.path.insert(0,'sleep_staging_deep_learning')
#from models import *


def myprint(epoch_status):
    sm = Counter(epoch_status)
    for k, v in sm.items():
        print(f'{k}: {v}/{len(epoch_status)}, {v*100./len(epoch_status)}%')


"""
def do_sleep_staging(epochs, sleep_stager, sleep_stager_hmm):
    epochs2 = epochs.transpose(0,2,1)
    sleep_stages = [sleep_stager.predict(epochs2[...,[x]], verbose=False) for x in range(epochs2.shape[-1])]
    sleep_stages = sum(sleep_stages)/len(sleep_stages)  # this model only input 1 channel, average predicted prob from all channels
    sleep_stages = np.argmax(sleep_stages, axis=1)

    sleep_stages = sleep_stager_hmm.predict(sleep_stages.reshape(-1,1))
    sleep_stages += 1 # to match to stage encoding

    # fix initial R to W
    for k,l in groupby(sleep_stages):
        ll = len(list(l))
        if k==4:
            sleep_stages[:ll] = 5
        break
    return sleep_stages
"""


if __name__=='__main__':
    epoch_length = 30 # [s]
    amplitude_thres = 500 # [uV]
    line_freq = 60.  # [Hz]
    bandpass_freq = [0.5, 20.]  # [Hz]
    n_jobs = 4
    clean_only = True
    newFs = 200.
    stages = ['W','N1','N2','N3','R']
    stage2num = {'W':5,'R':4,'N1':3,'N2':2,'N3':1}  # sleep staging encoding is different from ones on NSRR
    # if we run step2, we need to map
    num2stage = {stage2num[x]:x for x in stage2num}
    minimum_epochs_per_stage = 2

    # load sleep staging model  # if there is annotation file, use the annotation file
    #sleep_stager = conv_model(num_classes=5)
    #sleep_stager.load_weights('sleep_staging_deep_learning/model.hdf5')
    #sleep_stager_hmm = joblib.load('sleep_staging_deep_learning/hmm.joblib')

    # get list of files
    df = pd.read_excel('mastersheet.xlsx')
            
    # define output folder
    output_feature_dir = 'features'
    os.makedirs(output_feature_dir, exist_ok=True)
    output_sleep_vis_dir = 'sleep_spectrogram_hypnogram'
    os.makedirs(output_sleep_vis_dir, exist_ok=True)

    # for each recording
    for si in tqdm(range(len(df))):
        sid = df.SID.iloc[si]
        age = df.Age.iloc[si]
        sex = df.Sex.iloc[si]
        signal_path = df.SignalPath.iloc[si]
        annot_path  = df.AnnotPath.iloc[si]
        feature_path1 = os.path.join(output_feature_dir, f'features_{sid}.mat')
        feature_path2 = os.path.join(output_feature_dir, f'features_{sid}.csv')
        figure_path = os.path.join(output_sleep_vis_dir, sid+'.png')
        
        # load dataset
        EEG, sleep_stages, Fs, EEG_channels, combined_EEG_channels, combined_EEG_channels_ids, start_time = load_shhs_data(signal_path, annot_path)
        # this part should be customized according to how moonlight works
        # EEG is array of (#channels, # sample points) in uV
        # sleep_stages is array of (# sample points)
        # Fs is sampling rate in Hz
        # EEG_channels is array of channel names, assume EEG_channels = ['C3-M2', 'C4-M1']
        # combined_EEG_channels = ['C']
        # combined_EEG_channels_ids = [[0,1]]

        # segment EEG
        epochs, epoch_start_idx, epoch_status, specs, freq, qs = segment_EEG(EEG, epoch_length, epoch_length, Fs, newFs, notch_freq=line_freq, bandpass_freq=bandpass_freq, amplitude_thres=amplitude_thres, n_jobs=n_jobs)
        # epochs.shape = (#epochs, #channels, # sample points in one epoch)
        # the last output 'qs' is for amplitude normalization, it stands for 'quantiles' of signal amplitude, it contains q1 (25% percentile), q3 (75% percentile)
        # (signal - q1)/(q3-q1)
        if epochs.shape[0] <= 0:
            raise ValueError('Empty EEG segments')
        Fs = newFs  # in the segment_EEG function, the output epochs is resampled to 200Hz
        specs_db = 10*np.log10(specs)
        # specs_db is only for plotting

        # sleep staging
        #sleep_stages = do_sleep_staging(epochs, sleep_stager, sleep_stager_hmm)
        sleep_stages = sleep_stages[epoch_start_idx]
        # stage encoding:
        # W 5
        # R 4
        # N1 3
        # N2 2
        # N3 1

        # plot spectrogram and sleep stages
        # plot start
        plt.close()
        fig = plt.figure(figsize=(13.8,8.4))
        gs = fig.add_gridspec(1+len(combined_EEG_channels), 1, height_ratios=[1]+[3]*len(combined_EEG_channels))

        tt = np.arange(len(sleep_stages))*epoch_length/3600
        xticks = np.arange(0, np.floor(tt.max())+1) 
        xticklabels = []
        for j, x in enumerate(xticks):
            dt = start_time+datetime.timedelta(hours=x)
            xx = datetime.datetime.strftime(dt, '%H:%M:%S')#\n%m/%d/%Y')
            xticklabels.append(xx)
        
        ax_ss = fig.add_subplot(gs[0])
        ax_ss.step(tt, sleep_stages, color='k', where='post')
        ax_ss.yaxis.grid(True)
        ax_ss.set_ylim([0.7,5.3])
        ax_ss.set_yticks([1,2,3,4,5])
        ax_ss.set_yticklabels(['N3', 'N2', 'N1', 'R', 'W'])
        ax_ss.set_xlim([tt.min(), tt.max()])
        seaborn.despine()
        plt.setp(ax_ss.get_xlabel(), visible=False)
        plt.setp(ax_ss.get_xticklabels(), visible=False)
        
        for chi in range(len(combined_EEG_channels)):
            ax_spec = fig.add_subplot(gs[1+chi], sharex=ax_ss)
            specs_db_ch = (specs_db[:,chi*2]+specs_db[:,chi*2+1])/2
            ax_spec.imshow(
                    specs_db_ch.T, aspect='auto', origin='lower', cmap='jet',
                    vmin=-5, vmax=15,
                    extent=(tt.min(), tt.max(), freq.min(), freq.max()))
            ax_spec.text(-0.05, 0.5, f'Avg {combined_EEG_channels[chi]}',
                ha='center', va='center', transform=ax_spec.transAxes)
            #ax_spec.set_ylabel('freq (Hz)')
            ax_spec.set_xticks(xticks)
            ax_spec.set_xticklabels(xticklabels)
            if chi<len(combined_EEG_channels)-1:
                plt.setp(ax_spec.get_xlabel(), visible=False)
                plt.setp(ax_spec.get_xticklabels(), visible=False)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.11)
        plt.savefig(figure_path, bbox_inches='tight', pad_inches=0.03)
        # plot end

        if clean_only:
            good_ids = np.where(epoch_status=='clean')[0]
            if len(good_ids)<=300:
                myprint(epoch_status)
                raise ValueError('<=300 clean epochs')
            epochs = epochs[good_ids]
            specs = specs[good_ids]
            sleep_stages = sleep_stages[good_ids]
            epoch_start_idx = epoch_start_idx[good_ids]

        # extract brain age features

        # normalize signal
        nch = epochs.shape[1]
        q1, q2, q3 = qs
        epochs = (epochs - q2.reshape(1,nch,1)) / (q3.reshape(1,nch,1)-q1.reshape(1,nch,1))

        features, feature_names = extract_features(
            epochs, Fs, EEG_channels, 2,
            2, 1, return_feature_names=True,
            combined_channel_names=combined_EEG_channels,
            n_jobs=n_jobs, verbose=True)
        # features is a array/matrix (dataframe in R), shape = (#epochs, #features)
        # feature_names is a array of feature names
        artifact_ratio = 1-len(sleep_stages)/len(epoch_status)
        num_missing_stage = 5-len(set(sleep_stages[~np.isnan(sleep_stages)]))
        
        myprint(epoch_status)
        # save the mat file
        sio.savemat(feature_path1, {
            'start_time':start_time.strftime('%Y-%m-%d %H:%M:%S'),
            #'EEG_feature_names':feature_names,
            #'EEG_features':features,
            'EEG_channels':EEG_channels,
            'combined_EEG_channels':combined_EEG_channels,
            'combined_EEG_channels_ids':combined_EEG_channels_ids,
            'EEG_specs':specs,
            'EEG_frequency':freq,
            'sleep_stages':sleep_stages,
            'epoch_start_idx':epoch_start_idx,
            'age':age,
            'sex':sex,
            'epoch_status':epoch_status,
            'Fs':Fs,
            'artifact_ratio':artifact_ratio,
            'num_missing_stage':num_missing_stage,
            })
                
        # the next steps are to save the csv file

        # log-transform brain age features
        features_no_log = np.array(features)
        features = np.sign(features)*np.log1p(np.abs(features))
        
        # average features across sleep stages
        # the resulting feature always has fixed dimension,
        # if there is any missing stage, the features for that stage is NA
        # the NA values will be imputed using the training data (KNN imputation)
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
        
        # save X to .csv file
        cols = np.concatenate([[x.strip()+'_'+stage for x in feature_names] for stage in stages])
        df_feat = pd.DataFrame(data=X.reshape(1,-1), columns=cols)
        df_feat.insert(0, 'SID', sid)
        df_feat.to_csv(feature_path2, index=False)

        # save X_no_log to .csv file
        df_feat = pd.DataFrame(data=X_no_log.reshape(1,-1), columns=cols)
        df_feat.insert(0, 'SID', sid)
        df_feat.to_csv(feature_path2.replace('.csv','_no_log.csv'), index=False)


        # the output from step2 (to be used from step5 (compute BA)) is a csv file
        
