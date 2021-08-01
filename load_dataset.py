import numpy as np
import pandas as pd
import mne


def get_ApoE_brain_age_dir():
    return 'brain_age_model_fco'
    
def get_ApoE_channels():
    # make sure channel_names is always close to F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1, this is what is used in the brain age model
    return \
        ['F1-A2', 'Fz-A2', 'C3-A2', 'C4-A1', 'O1-x', 'O2-x'],\
        ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1'],\
        [[0,1],[2,3],[4,5]],\
        ['F', 'C', 'O']
    
def load_ApoE_dataset(signal_path, annot_path, epoch_sec=30):
    # load signal
    edf = mne.io.read_raw_edf(signal_path, preload=False, verbose=False, stim_channel=None)
    Fs = edf.info['sfreq']
    start_time = edf.info['meas_date'].replace(tzinfo=None)
    #all_ch_names = edf.info['ch_names']
    
    # assumes EEG_channels = [F3M2, F4M1, C3M2, C4M1, O1M2, O2M1]
    # when computing features, the spectral features were averaged across left and right
    # combined_EEG_channels is to define the name of the averaged features
    ch_names, standard_ch_names, pair_ch_ids, combined_ch_names = get_ApoE_channels()
    signals = edf.get_data(picks=ch_names)
    
    # mne automatically converts to V, convert back to uV
    #signals *= 1e6
    # no, for ApoE, seems like the signal is already in uV
    
    # load annotation
    annot = pd.read_csv(annot_path, header=None, sep='\s+')
    assert np.all(np.diff(annot[0])==1)
    
    sleep_stage_mapping = {
        7:np.nan, # unscored
        5:4, # R
        4:1, # N3
        3:1, # N3
        2:2, # N2
        1:3, # N1
        0:5, #W
    }
    sleep_stages = [sleep_stage_mapping[x] for x in annot[1]]
    
    # make sure sleep stages and signals have the same length
    epoch_size = int(round(epoch_sec*Fs))
    if len(sleep_stages)*epoch_size<=signals.shape[1]:
        sleep_stages = np.repeat(sleep_stages, epoch_size)
        signals = signals[:,:len(sleep_stages)]
    elif len(sleep_stages)*epoch_size>signals.shape[1]:
        num_epoch = signals.shape[1]//epoch_size
        sleep_stages = np.repeat(sleep_stages[:num_epoch], epoch_size)
        signals = signals[:,:num_epoch*epoch_size]
    
    return signals, sleep_stages, ch_names, combined_ch_names, Fs, start_time
    
        
def get_STAGES_brain_age_dir():
    return 'brain_age_model_fco'
    
def get_STAGES_channels():
    return \
        ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1'],\
        ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1'],\
        [[0,1],[2,3],[4,5]],\
        ['F', 'C', 'O']
    
def load_STAGES_dataset(signal_path, annot_path, epoch_sec=30):
    # load signal
    edf = mne.io.read_raw_edf(signal_path, preload=False, verbose=False, stim_channel=None)
    Fs = edf.info['sfreq']
    start_time = edf.info['meas_date'].replace(tzinfo=None)
    #all_ch_names = edf.info['ch_names']
    
    # TODO make sure channel_names is always close to F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1, this is what is used in the brain age model
    ch_names = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'M1', 'M2']
    signals = edf.get_data(picks=ch_names)
    signals = np.array([
        signals[ch_names.index('F3')] - signals[ch_names.index('M2')],
        signals[ch_names.index('F4')] - signals[ch_names.index('M1')],
        signals[ch_names.index('C3')] - signals[ch_names.index('M2')],
        signals[ch_names.index('C4')] - signals[ch_names.index('M1')],
        signals[ch_names.index('O1')] - signals[ch_names.index('M2')],
        signals[ch_names.index('O2')] - signals[ch_names.index('M1')],
        ])
    # assumes EEG_channels = [F3M2, F4M1, C3M2, C4M1, O1M2, O2M1]
    # when computing features, the spectral features were averaged across left and right
    # combined_EEG_channels is to define the name of the averaged features
    ch_names, standard_ch_names, pair_ch_ids, combined_ch_names = get_STAGES_channels()
    
    # mne automatically converts to V, convert back to uV
    #signals *= 1e6
    # no, for STAGES, seems like the signal is already in uV
    
    # load annotation
    annot = pd.read_csv(annot_path)
    annot['Start Time'] = pd.to_datetime(annot['Start Time'])
    annot['Event'] = annot.Event.astype(str).str.strip().str.lower()
    annot = annot[np.in1d(annot.Event, ['wake', 'rem', 'stage1', 'stage2', 'stage3'])].reset_index(drop=True)
    
    sleep_stage_mapping = {
        'wake':5,
        'rem':4,
        'stage1':3,
        'stage2':2,
        'stage3':1,
    }   
    sleep_stages = np.zeros(signals.shape[1])+np.nan
    
    for i in range(len(annot)):
        start = (annot['Start Time'].iloc[i]-start_time).total_seconds()
        end = start + annot['Duration (seconds)'].iloc[i]
        start = int(round(start*Fs))
        end = int(round(end*Fs))
        start = max(0, start)
        end = min(len(sleep_stages), end)
        sleep_stages[start:end] = sleep_stage_mapping[annot.Event.iloc[i]]
    
    return signals, sleep_stages, ch_names, combined_ch_names, Fs, start_time


def get_WSC_brain_age_dir():
    return 'brain_age_model_c'
    
def get_WSC_channels():
    return \
        ['C3-x', 'C4-x'],\
        ['C3-M2', 'C4-M1'],\
        [[0,1]],\
        ['C']
    
def load_WSC_dataset(signal_path, annot_path, epoch_sec=30):
    # load signal
    edf = mne.io.read_raw_edf(signal_path, preload=False, verbose=False, stim_channel=None)
    Fs = edf.info['sfreq']
    start_time = edf.info['meas_date'].replace(tzinfo=None)
    #all_ch_names = edf.info['ch_names']
    
    # when computing features, the spectral features were averaged across left and right
    # combined_EEG_channels is to define the name of the averaged features
    ch_names, standard_ch_names, pair_ch_ids, combined_ch_names = get_WSC_channels()
    signals = edf.get_data(picks=ch_names[0])  # specific to WSC
    signals = np.vstack([signals,signals])
    
    # mne automatically converts to V, convert back to uV
    signals *= 1e6
    
    # load annotation
    annot = pd.read_csv(annot_path, header=None, sep='\s+')
    assert np.all(np.diff(annot[0])==1)
    
    sleep_stage_mapping = {
        7:np.nan, # unscored
        5:4, # R
        4:1, # N3
        3:1, # N3
        2:2, # N2
        1:3, # N1
        0:5, #W
    }
    sleep_stages = [sleep_stage_mapping[x] for x in annot[1]]
    
    # make sure sleep stages and signals have the same length
    epoch_size = int(round(epoch_sec*Fs))
    if len(sleep_stages)*epoch_size<=signals.shape[1]:
        sleep_stages = np.repeat(sleep_stages, epoch_size)
        signals = signals[:,:len(sleep_stages)]
    elif len(sleep_stages)*epoch_size>signals.shape[1]:
        num_epoch = signals.shape[1]//epoch_size
        sleep_stages = np.repeat(sleep_stages[:num_epoch], epoch_size)
        signals = signals[:,:num_epoch*epoch_size]
    
    return signals, sleep_stages, ch_names, combined_ch_names, Fs, start_time
    
