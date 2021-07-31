import subprocess
import sys
import numpy as np
import scipy.stats as stats
from scipy.signal import detrend
from joblib import Parallel, delayed
#import nitime.algorithms as tsa
from multitaper_spectrogram import *
from bandpower import *
# this is python version of sample entropy which is very slow but runs on different OSs
#from sampen_python.sampen2 import sampen2  # https://sampen.readthedocs.io/en/latest/
# this is C++ version of sample entropy which is not slow but needs compilation on Linux
#SAMPEN_PATH = 'sampen_C/sampen'  # https://www.physionet.org/physiotools/sampen/


def compute_features_each_seg(eeg_seg, Fs, NW, band_freq, band_names, total_freq_range, combined_channel_names=None, window_length=None, window_step=None, need_sampen=False):
    """
    Compute features for each segment
    eeg_seg: np.array, shape=(#channel, #sample points)
    Fs: sampling frequency in Hz
    NW: time-halfbandwidth product to control the freq resolution, NW = bandwidth/2 * window_time
    band_freq: list of the frequencies of the bands, such as [[0.5,4],[4,8]] for delta and theta
    band_names: list of band names, must have same length as band_freq
    total_freq_range: list, the total freq range for relative power, such as [0.5,20] for sleep
    combined_channel_names: optional, for sleep montage F3M2, F4M1, C3M2, C4M1, O1M2, O2M1 only, set to either None (default) or ['F','C','O']
    window_length: window size for time-freq spectrogram inside this segment, specified in # of sample points. Default is None which equals to eeg_seg.shape[-1]
    window_length: window step for time-freq spectrogram inside this segment, specified in # of sample points. Default is None which equals to eeg_seg.shape[-1]
    need_sampen: bool, whether to compute sample entropy, which is slow. Default to False.
    """
    assert len(band_freq)==len(band_names), 'band_names must have same length as band_freq'
    if window_length is None or window_step is None:
        window_length = eeg_seg.shape[-1]
        window_step = eeg_seg.shape[-1]
        
    # compute spectrogram
    # spec, shape=(window_num, freq_point_num, channel_num)
    # freq, shape=(freq_point_num,)
    spec, freq = multitaper_spectrogram(eeg_seg, Fs, NW, window_length, window_step)
    
    if combined_channel_names is not None:
        # TODO assume left and right, left and right, ... channels
        spec = (spec[:,:,::2]+spec[:,:,1::2])/2.0

    # band power
    bp, band_findex = bandpower(spec, freq, band_freq, total_freq_range=total_freq_range, relative=False)

    ## time domain features
    
    # signal line length
    f1 = np.abs(np.diff(eeg_seg,axis=1)).sum(axis=1)*1.0/eeg_seg.shape[-1]
    # signal kurtosis
    f2 = stats.kurtosis(eeg_seg,axis=1,nan_policy='propagate')
    if need_sampen:
        # signal sample entropy
        f3 = []
        for ci in range(len(eeg_seg)):
            #Bruce, E. N., Bruce, M. C., & Vennelaganti, S. (2009).
            #Sample entropy tracks changes in EEG power spectrum with sleep state and aging. Journal of clinical neurophysiology, 26(4), 257.
            # python version (slow, for multiple OSs)
            f3.append(sampen2(list(eeg_seg[ci]),mm=2,r=0.2,normalize=True)[-1][1])  # sample entropy
            # C++ version (not slow, for Linux only)
            #sp = subprocess.Popen([SAMPEN_PATH,'-m','2','-r','0.2','-n'],stdout=subprocess.PIPE,stdin=subprocess.PIPE,stderr=subprocess.STDOUT)#
        
    ## frequency domain features
    
    f4 = [];f5 = [];f6 = [];f7 = [];f9 = []
    band_num = len(band_freq)
    for bi in range(band_num):
        if band_names[bi].lower()!='sigma': # no need for sigma band
            if len(spec)>1:  # this segment is split into multiple sub-windows
                # max, min, std of band power inside this segment
                f4.extend(np.percentile(bp[bi],95,axis=0))
                f5.extend(bp[bi].min(axis=0))
                f7.extend(bp[bi].std(axis=0))
            # mean band power inside this segment
            f6.extend(bp[bi].mean(axis=0))

        if len(spec)>1:
            # spectral kurtosis as a rough density measure of transient events such as spindle
            spec_flatten = spec[:,band_findex[bi],:].reshape(spec.shape[0]*len(band_findex[bi]),spec.shape[2])
            f9.extend(stats.kurtosis(spec_flatten, axis=0, nan_policy='propagate'))

    f10 = []
    delta_theta = bp[0]/(bp[1]+1)
    if len(spec)>1:  # this segment is split into multiple sub-windows
        # max, min, std, mean of delta/theta ratios inside this segment
        f10.extend(np.percentile(delta_theta,95,axis=0))
        f10.extend(np.min(delta_theta,axis=0))
    f10.extend(np.mean(delta_theta,axis=0))
    if len(spec)>1:
        f10.extend(np.std(delta_theta,axis=0))
    
    f11 = []
    delta_alpha = bp[0]/(bp[2]+1)
    if len(spec)>1:  # this segment is split into multiple sub-windows
        # max, min, std, mean of delta/alpha ratios inside this segment
        f11.extend(np.percentile(delta_alpha,95,axis=0))
        f11.extend(np.min(delta_alpha,axis=0))
    f11.extend(np.mean(delta_alpha,axis=0))
    if len(spec)>1:
        f11.extend(np.std(delta_alpha,axis=0))
    
    f12 = []
    theta_alpha = bp[1]/(bp[2]+1)
    # max, min, std, mean of theta/alpha ratios inside this segment
    if len(spec)>1:  # this segment is split into multiple sub-windows
        f12.extend(np.percentile(theta_alpha,95,axis=0))
        f12.extend(np.min(theta_alpha,axis=0))
    f12.extend(np.mean(theta_alpha,axis=0))
    if len(spec)>1:
        f12.extend(np.std(theta_alpha,axis=0))

    if need_sampen:
        return np.r_[f1,f2,f3,f4,f5,f6,f7,f9,f10,f11,f12]
    else:
        return np.r_[f1,f2,f4,f5,f6,f7,f9,f10,f11,f12]


def extract_features(eeg_segs, Fs, channel_names, NW, sub_window_time=None, sub_window_step=None, return_feature_names=True, combined_channel_names=None, need_sampen=False, n_jobs=1, verbose=True):
    """
    Extract features from EEG segments in parallel.

    Arguments:
    eeg_segs: np.ndarray, shape=(#seg, #channel, #sample points)
    Fs: sampling frequency in Hz
    channel_names: a list of channel names
    NW: time-halfbandwidth product to control the freq resolution, NW = bandwidth/2 * window_time
    sub_window_time: sub-window time in seconds for spectrogram inside each segment. Default is None which equals to eeg_seg.shape[-1]
    window_length: window step for time-freq spectrogram inside this segment, specified in # of sample points. Default is None which equals to eeg_seg.shape[-1]
    return_feature_names: bool, if to return feature names as a list, default True.
    combined_channel_names: optional, for sleep montage F3M2, F4M1, C3M2, C4M1, O1M2, O2M1 only, set to either None (default) or ['F','C','O']
    need_sampen: bool. need to compute sample entropy. default is False
    n_jobs: number of CPUs to run in parallel, default is 1 (serial), set to -1 to use all CPUs

    Outputs:
    features from each segment in np.ndarray type, shape=(#seg, #feature)
    a list of names of each feature
    """

    seg_num = len(eeg_segs)
    if seg_num <= 0:
        return []

    band_names = ['delta','theta','alpha','sigma']
    band_freq = [[0.5,4],[4,8],[8,12],[12,20]]  # [Hz]
    tostudy_freq = [0.5, 20.]  # [Hz]

    sub_window_size = int(round(sub_window_time*Fs))
    sub_step_size = int(round(sub_window_step*Fs))
    
    #old_threshold = np.get_printoptions()['threshold']
    #np.set_printoptions(threshold=np.nan)
    with Parallel(n_jobs=n_jobs, verbose=verbose) as parallel:
        features = parallel(delayed(compute_features_each_seg)(
                eeg_segs[segi], Fs, NW,
                band_freq, band_names, tostudy_freq,
                combined_channel_names,
                sub_window_size, sub_step_size, need_sampen) for segi in range(seg_num))
    #np.set_printoptions(threshold=old_threshold)

    if return_feature_names:
        feature_names = ['line_length_%s'%chn for chn in channel_names]
        feature_names += ['kurtosis_%s'%chn for chn in channel_names]
        if need_sampen:
            feature_names += ['sample_entropy_%s'%chn for chn in channel_names]

        if sub_window_time is None or sub_window_step is None:
            feats = ['mean']
        else:
            feats = ['max','min','mean','std','kurtosis']
        for ffn in feats:
            for bn in band_names:
                if ffn=='kurtosis' or bn!='sigma': # no need for sigma band
                    feature_names += ['%s_bandpower_%s_%s'%(bn,ffn,chn) for chn in combined_channel_names]

        power_ratios = ['delta/theta','delta/alpha','theta/alpha']
        for pr in power_ratios:
            if not (sub_window_time is None or sub_window_step is None):
                feature_names += ['%s_max_%s'%(pr,chn) for chn in combined_channel_names]
                feature_names += ['%s_min_%s'%(pr,chn) for chn in combined_channel_names]
            feature_names += ['%s_mean_%s'%(pr,chn) for chn in combined_channel_names]
            if not (sub_window_time is None or sub_window_step is None):
                feature_names += ['%s_std_%s'%(pr,chn) for chn in combined_channel_names]

    if return_feature_names:
        return np.array(features), feature_names#, pxx_mts, freqs
    else:
        return np.array(features)#, pxx_mts, freqs

