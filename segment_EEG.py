import numpy as np
from scipy.signal import detrend
from scipy.stats import mode
#from joblib import Parallel, delayed
from mne.filter import filter_data, notch_filter
#from scikits.samplerate import resample
from scipy.signal import resample
from mne.time_frequency import psd_array_multitaper


epoch_status_explanation = [
    'clean',
    'NaN in sleep stage',
    'NaN in EEG',
    'overly high/low amplitude',
    'flat signal']


def segment_EEG(EEG, window_time, step_time, Fs, newFs=200, notch_freq=None, bandpass_freq=None, start_end_remove_window_num=0, amplitude_thres=500, n_jobs=1, to_remove_mean=False):#
    """Segment EEG signals.

    Arguments:
    EEG -- np.ndarray, size=(channel_num, sample_num)
    window_time -- in seconds
    step_time -- in seconds
    Fs -- in Hz

    Keyword arguments:
    newFs -- sfreq to be resampled
    notch_freq
    bandpass_freq
    start_end_remove_window_num -- default 0, number of windows removed at the beginning and the end of the EEG signal
    amplitude_thres -- default 500, mark all segments with np.any(EEG_seg>=amplitude_thres)=True
    to_remove_mean -- default False, whether to remove the mean of EEG signal from each channel
    """
    std_thres = 0.2
    std_thres2 = 1.
    flat_seconds = 5
    padding = 0
    
    if to_remove_mean:
        EEG = EEG - np.nanmean(EEG,axis=1, keepdims=True)
    window_size = int(round(window_time*Fs))
    step_size = int(round(step_time*Fs))
    flat_length = int(round(flat_seconds*Fs))
    
    start_ids = np.arange(0, EEG.shape[1]-window_size+1, step_size)
    if start_end_remove_window_num>0:
        start_ids = start_ids[start_end_remove_window_num:-start_end_remove_window_num]
    
    # first assign clean to all epoch status
    epoch_statuss = [epoch_status_explanation[0]]*len(start_ids)
    
    if notch_freq is not None and Fs/2>notch_freq:# and bandpass_freq is not None and np.max(bandpass_freq)>=notch_freq:
        EEG = notch_filter(EEG, Fs, notch_freq, fir_design="firwin", verbose=False)  # (#window, #ch, window_size+2padding)
    if bandpass_freq is None:
        fmin = None
        fmax = None
    else:
        fmin = bandpass_freq[0]
        fmax = bandpass_freq[1]
    if fmax>=Fs/2:
        fmax = None
    if bandpass_freq is not None:
        EEG = filter_data(EEG, Fs, fmin, fmax, fir_design="firwin", verbose=False)#detrend(EEG, axis=1), n_jobs='cuda'
    
    # resample
    if Fs!=newFs:
        #r = newFs*1./Fs
        #EEG = Parallel(n_jobs=n_jobs, verbose=False)(delayed(resample)(EEG[i], r, 'sinc_best') for i in range(len(EEG)))
        #EEG = np.array(EEG).astype(float)
        Nnew = int(round(EEG.shape[1]/Fs*newFs))
        EEG = resample(EEG, Nnew, axis=1)
        Fs = newFs
        window_size = int(round(window_time*Fs))
        step_size = int(round(step_time*Fs))
        flat_length = int(round(flat_seconds*Fs))
        start_ids = np.arange(0, EEG.shape[1]-window_size+1, step_size)
        if start_end_remove_window_num>0:
            start_ids = start_ids[start_end_remove_window_num:-start_end_remove_window_num]
    
    # get quantiles
    EEG2 = np.array(EEG)
    EEG2[np.abs(EEG2)<1e-5] = np.nan
    qs = np.nanpercentile(EEG2, (25,50,75), axis=1)
    
    #segment into epochs
    EEG_segs = EEG[:,list(map(lambda x:np.arange(x-padding,x+window_size+padding), start_ids))].transpose(1,0,2)  # (#window, #ch, window_size+2padding)
    
    #TODO detrend(EEG_segs)
    #TODO remove_mean(EEG_segs) to remove frequency at 0Hz
    
    NW = 10.
    BW = NW*2./window_time
    if fmin is None:
        fmin = 0
    if fmax is None:
        fmax = np.inf
    specs, freq = psd_array_multitaper(EEG_segs, Fs, fmin=fmin, fmax=fmax, adaptive=False, low_bias=True, n_jobs=n_jobs, verbose='ERROR', bandwidth=BW, normalization='full')
    
    nan2d = np.any(np.isnan(EEG_segs), axis=2)
    nan1d = np.where(np.any(nan2d, axis=1))[0]
    for i in nan1d:
        epoch_statuss[i] = epoch_status_explanation[2]
    
    # EEG_segs.shape = (#epochs, 6, 6000)
    amplitude_large2d = np.any(np.abs(EEG_segs)>amplitude_thres, axis=2)  #(#epoch, 6)
    amplitude_large1d = np.where(np.any(amplitude_large2d, axis=1))[0] #(#epoch,)
    for i in amplitude_large1d:
        epoch_statuss[i] = epoch_status_explanation[3]
          
    # if there is any flat signal with flat_length
    short_segs = EEG_segs.reshape(EEG_segs.shape[0], EEG_segs.shape[1], EEG_segs.shape[2]//flat_length, flat_length)
    flat2d = np.any(detrend(short_segs, axis=3).std(axis=3)<=std_thres, axis=2)
    flat2d = np.logical_or(flat2d, np.std(EEG_segs,axis=2)<=std_thres2)
    flat1d = np.where(np.any(flat2d, axis=1))[0]
    for i in flat1d:
        epoch_statuss[i] = epoch_status_explanation[4]
     
    lens = [len(EEG_segs), len(start_ids), len(epoch_statuss), len(specs)]
    if len(set(lens))>1:
        minlen = min(lens)
        EEG_segs = EEG_segs[:minlen]
        start_ids = start_ids[:minlen]
        epoch_statuss = epoch_statuss[:minlen]
        specs = specs[:minlen]

    return EEG_segs, start_ids, np.array(epoch_statuss), specs, freq, qs


