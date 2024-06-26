import datetime
import re
import numpy as np
import xmltodict
import mne


def load_shhs_data(signal_path, annot_path):
    edf = mne.io.read_raw_edf(signal_path, stim_channel=None, preload=False, verbose=False)
    eeg = edf.get_data(picks=['EEG(sec)', 'EEG'])
    Fs = edf.info['sfreq']
    eeg_channels = ['C3-M2', 'C4-M1']
    start_time = edf.info['meas_date'].replace(tzinfo=None)
    combined_EEG_channels = ['C']
    combined_EEG_channels_ids = [[0,1]]

    # convert to uV
    eeg = eeg*1e6

    # get sleep stages
    mapping = {
        0:5,  # W/W
        1:3,  # N1/S1
        2:2,  # N2/S2
        3:1,  # N3/S3
        4:1,  # N3/S4
        5:4,  # R/REM
            }
    with open(annot_path,'r') as ff:
        annot = xmltodict.parse(ff.read())
    ss = np.array(annot['CMPStudyConfig']['SleepStages']['SleepStage']).astype(int)
    epoch_time = float(annot['CMPStudyConfig']['EpochLength'])
    epoch_size = int(round(epoch_time*Fs))
    sleep_stages = np.zeros(eeg.shape[1])+np.nan
    for i, s in enumerate(ss):
        sleep_stages[i*epoch_size:(i+1)*epoch_size] = mapping.get(s, np.nan)

    return eeg, sleep_stages, Fs, eeg_channels, combined_EEG_channels, combined_EEG_channels_ids, start_time



def load_bidmc_edf(path):
    edf = mne.io.read_raw_edf(path, stim_channel=None, preload=False, verbose=False)
    edf_channels = edf.info['ch_names']
    Fs = edf.info['sfreq']
    if type(edf.info['meas_date'])==tuple:
        start_time = datetime.datetime.fromtimestamp(edf.info['meas_date'][0])+ timedelta(seconds=time.altzone)
    else:
        start_time = edf.info['meas_date']

    # find EEG channels
    # case 1: contralateral montage
    eeg_channels_regex_left  = ['^F3-(A|M)', '^C3-(A|M)', '^O1-(A|M)']
    eeg_channels_regex_right = ['^F4-(A|M)', '^C4-(A|M)', '^O2-(A|M)']
    all_ok_channels_left  = [[x for x in edf_channels if re.match(regex, x)] for regex in eeg_channels_regex_left]
    all_ok_channels_right = [[x for x in edf_channels if re.match(regex, x)] for regex in eeg_channels_regex_right]
    case1 = all([len(x1)+len(x2)>=1 for x1, x2 in zip(all_ok_channels_left, all_ok_channels_right)])
    if case1:
        eeg_channels = []
        for i in range(3):
            if len(all_ok_channels_left[i])>0 and len(all_ok_channels_right[i])>0:
                # use left and right
                eeg_channels.append(all_ok_channels_left[i][0])
                eeg_channels.append(all_ok_channels_right[i][0])
            elif len(all_ok_channels_left[i])>0 and len(all_ok_channels_right[i])==0:
                # use both left
                eeg_channels.append(all_ok_channels_left[i][0])
                eeg_channels.append(all_ok_channels_left[i][0])
            elif len(all_ok_channels_left[i])==0 and len(all_ok_channels_right[i])>0:
                # use both right
                eeg_channels.append(all_ok_channels_right[i][0])
                eeg_channels.append(all_ok_channels_right[i][0])
        eeg = edf.get_data(picks=eeg_channels)  # eeg.shape=(#channel, T)
    else: # test case 2
        eeg_channels_regex = ['^(A|M)1$', '^(A|M)2$', '^C3$', '^C4$', '^O1$', '^O2$', '^F3$', '^F4$']
        all_ok_channels = [[x for x in edf_channels if re.match(regex, x)] for regex in eeg_channels_regex]
        case2 = all([len(x)>=1 for x in all_ok_channels])
        if case2:
            eeg_channels = [x[0] for x in all_ok_channels]
            eeg = edf.get_data(picks=eeg_channels)
            eeg = np.array([eeg[6]-eeg[1],  # F3-M2
                            eeg[7]-eeg[0],  # F4-M1
                            eeg[2]-eeg[1],  # C3-M2
                            eeg[3]-eeg[0],  # C4-M2
                            eeg[4]-eeg[1],  # O1-M2
                            eeg[5]-eeg[0],])# O2-M2
        else:
            raise Exception(f'Channel name format is wrong! Found {edf_channels}')
        
    # convert to uV
    eeg = eeg*1e6

    eeg_channels = ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1']
    return eeg, Fs, eeg_channels, ['F', 'C', 'O'], [[0,1],[2,3],[4,5]], start_time
    
