mport datetime
from dateutil.parser import parse
import os
import glob
import pickle
import subprocess
import shutil
import numpy as np
import scipy.io as sio
import pandas as pd
import h5py
import pyedflib
import mne
from tqdm import tqdm
#from tqdm.notebook import tqdm


def convert_to_edf(input_path, output_path, channels=None):
    """
    """
    # edf
    if input_path.endswith('.edf'):
        ff = mne.io.read_raw_edf(input_path, preload=True, verbose=False, stim_channel=None, exclude=['Fp1', 'Fp2', 'P3', 'P4', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Chin1', 'Chin2', 'Chin3', 'E2', 'E1', 'AirFlow', 'Chest', 'LAT', 'Snore', 'Abdomen', 'RAT', '30', 'IC', 'Leak', 'EtCO2', 'CO2 Wave', 'DC6', 'DC7', 'DC8', 'DC9', 'DC10', 'DC11', 'PTAF', 'SpO2', 'PR', 'Pleth', 'EDF Annotations'])
        signals = ff.get_data()
        channel_names = ff.info['ch_names']
        EEG = np.array([signals[channel_names.index('F3')] - signals[channel_names.index('M2')],
                        signals[channel_names.index('F4')] - signals[channel_names.index('M1')],
                        signals[channel_names.index('C3')] - signals[channel_names.index('M2')],
                        signals[channel_names.index('C4')] - signals[channel_names.index('M1')],
                        signals[channel_names.index('O1')] - signals[channel_names.index('M2')],
                        signals[channel_names.index('O2')] - signals[channel_names.index('M1')],])
        EEG = EEG*1e6  # mne.io.read_raw_edf automatically converts to V, we convert back to uV
        channel_names = ['F3M2','F4M1','C3M2','C4M1','O1M2','O2M1']
        if type(ff.info['meas_date'])==tuple:
            starttime = datetime.datetime.utcfromtimestamp(ff.info['meas_date'][0])
        elif type(ff.info['meas_date'])==int:
            starttime = datetime.datetime.utcfromtimestamp(ff.info['meas_date'])
        elif type(ff.info['meas_date'])==datetime.datetime:
            starttime = ff.info['meas_date']
        Fs = ff.info['sfreq']
        channel_info = [
                {'label': channel_names[i],
                 'dimension': 'uV',
                 'sample_rate': Fs,
                 'physical_max': 32767,
                 'physical_min': -32768,
                 'digital_max': 32767,
                 'digital_min': -32768,
                 'transducer': 'E',
                 'prefilter': ''}
            for i in range(len(channel_names))]

    # mat
    elif input_path.endswith('.mat'):

        try:
            # old mat format
            
            ff = sio.loadmat(input_path)
            channel_names = [ff['hdr'][0,i]['signal_labels'][0] for i in range(ff['hdr'].shape[1])]
            channel_names = [x.upper().replace('-','').replace('A','M').replace('C3M1','C3M2').replace('C4M2','C4M1').replace('F3M1','F3M2').replace('F4M2','F4M1').replace('O1M1','O1M2').replace('O2M2','O2M1') for x in channel_names]
            
            if channels is None:
                channel_ids = np.arange(ff['hdr'].shape[1])
            else:
                channel_ids = [channel_names.index(ch) for ch in channels]

            if 'Fs' not in ff:
                Fs = 200.

            channel_info = [
                    {'label': channel_names[i],
                     'dimension': ff['hdr'][0,i]['physical_dimension'][0],
                     'sample_rate': Fs,
                     'physical_max': ff['hdr'][0,i]['physical_max'][0,0],
                     'physical_min': ff['hdr'][0,i]['physical_min'][0,0],
                     'digital_max': ff['hdr'][0,i]['digital_max'][0,0],
                     'digital_min': ff['hdr'][0,i]['digital_min'][0,0],
                     'transducer': ff['hdr'][0,i]['tranducer_type'][0],
                     'prefilter': ff['hdr'][0,i]['prefiltering'][0] if len(ff['hdr'][0,i]['prefiltering'])>0 else ''}
                for i in channel_ids]

            EEG = ff['s'][channel_ids]
            
        except Exception as ee:
            # new mat format
            with h5py.File(input_path, 'r') as ff:
                channel_names_ref = ff['hdr']['signal_labels'][()].flatten()
                channel_names = [''.join(map(chr,ff[x][()].flatten())) for x in channel_names_ref]
                channel_names = [x.upper().replace('-','').replace('A','M').replace('C3M1','C3M2').replace('C4M2','C4M1').replace('F3M1','F3M2').replace('F4M2','F4M1').replace('O1M1','O1M2').replace('O2M2','O2M1') for x in channel_names]
                if channels is None:
                    channel_ids = np.arange(ff['hdr'].shape[1])
                else:
                    channel_ids = [channel_names.index(ch) for ch in channels]
            
                if 'physical_dimension' in ff['hdr']:
                    physical_dimension_ref = ff['hdr']['physical_dimension'][()].flatten()
                    physical_dimension = [''.join(map(chr,ff[physical_dimension_ref[x]][()].flatten())) for x in channel_ids]
                else:
                    physical_dimension = ['uV']*len(channel_ids)
                physical_max = [32767]*len(channel_ids)
                physical_min = [-32768]*len(channel_ids)
                digital_max = [32767]*len(channel_ids)
                digital_min = [-32768]*len(channel_ids)
                if 'tranducer_type' in ff['hdr']:
                    tranducer_type_ref = ff['hdr']['tranducer_type'][()].flatten()
                    tranducer_type = [''.join(map(chr,ff[tranducer_type_ref[x]][()].flatten())) for x in channel_ids]
                else:
                    tranducer_type = ['E']*len(channel_ids)
                prefilter = ['']*len(channel_ids)

                if 'recording' in ff:
                    Fs = float(ff['recording']['samplingrate'][0,0])
                else:
                    Fs = 200.

                EEG = np.array([ff['s'][:,x] for x in channel_ids])

            channel_info = [
                    {'label': channel_names[channel_ids[i]],
                     'dimension': physical_dimension[i],
                     'sample_rate': Fs,
                     'physical_max': physical_max[i],
                     'physical_min': physical_min[i],
                     'digital_max': digital_max[i],
                     'digital_min': digital_min[i],
                     'transducer': tranducer_type[i],
                     'prefilter': prefilter[i]}
                for i in range(len(channel_ids))]
        starttime = None
        
    if EEG.shape[1]<=100:
        raise ValueError('Short EEG')

    with pyedflib.EdfWriter(output_path, len(EEG), file_type=pyedflib.FILETYPE_EDFPLUS) as ff:
        ff.setSignalHeaders(channel_info)
        ff.writeSamples(EEG)
            
    return EEG.shape[1], Fs, starttime


def convert_to_xml(input_path, output_path, signal_len, Fs, starttime):
    """
    """
    if input_path.endswith('.csv'):
        stage_txt2num = {'w':5, 'awake':5,
                         'r':4, 'rem':4,
                         'n1':3, 'nrem1':3,
                         'n2':2, 'nrem2':2,
                         'n3':1, 'nrem3':1,}
        ss_df = pd.read_csv(input_path)
        ss_df.event = ss_df.event.str.lower().str.strip()
        ss_df = ss_df[ss_df.event.str.startswith('sleep_stage_')].reset_index(drop=True)
        ss_df['stage'] = ss_df.event.str.split('_', expand=True)[2]
        ss_df = ss_df[np.in1d(ss_df['stage'], list(stage_txt2num.keys()))].reset_index(drop=True)
                         
        oneday = datetime.timedelta(days=1)
        sleep_stages = np.zeros(signal_len)+np.nan
        for i in range(len(ss_df)):
            thistime = parse(ss_df.time.iloc[i], default=starttime)
            if thistime.hour<12 and starttime.hour>12:
                thistime += oneday
            assert thistime>=starttime, 'thistime<starttime'
            startid = round((thistime - starttime).total_seconds()*Fs)
            if startid<len(sleep_stages):
                sleep_stages[startid:] = stage_txt2num.get(ss_df.stage.iloc[i], np.nan)
            if i==len(ss_df)-1 and startid+round(Fs*30)<len(sleep_stages):
                sleep_stages[startid+round(Fs*30):] = np.nan
    
    elif input_path.endswith('.mat'):
        with h5py.File(input_path, 'r') as ff:
            sleep_stages = ff['stage'][:].flatten()
            
    sleep_stages = sleep_stages[np.arange(0, len(sleep_stages), int(round(Fs*30)))]
    sleep_stages[np.isnan(sleep_stages)] = -1
    sleep_stage_mapping = {-1:0, 0:0, 5:0, 4:5, 3:1, 2:2, 1:3}
    with open(output_path, 'w') as ff:
        ff.write('<CMPStudyConfig>\n')
        ff.write('<EpochLength>30</EpochLength>\n')
        ff.write('<SleepStages>\n')
        for ss in sleep_stages:
            ff.write('<SleepStage>%d</SleepStage>\n'%sleep_stage_mapping[ss])
        ff.write('</SleepStages>\n')
        ff.write('</CMPStudyConfig>')


def run_luna(output_dir, edf_paths, xml_paths, channels, stage, cfreq=13.5, cycles=12):
    # create the list file
    list_path = os.path.join(output_dir, 'luna.lst')
    df = pd.DataFrame(data={'sid': [os.path.basename(x).lower().replace('.edf','') for x in edf_paths], 'edf':edf_paths, 'xml':xml_paths})
    df = df[['sid', 'edf', 'xml']]
    df.to_csv(list_path, sep='\t', index=False, header=False)
    
    # generate R code to convert luna output .db to .mat
    db_path = os.path.join(output_dir, 'luna_output.db')
    xls_path = os.path.join(output_dir, 'luna_output.xlsx')
    r_code = """library(luna)
library(xlsx)
k<-ldb("%s")
d1<-lx(k,"SPINDLES", "CH_F")
d2<-lx(k,"SPINDLES", "CH")
d3 <- merge(d1,d2,by=c("ID","CH"))
write.xlsx(d3, "%s") 
"""%(db_path, xls_path)
    r_code_path = os.path.join(output_dir, 'convert_luna_output_db2mat.R')
    with open(r_code_path, 'w') as ff:
        ff.write(r_code)
    # maximum spindle duration 10s to account for pediatric subjects
    #with open(os.devnull, 'w') as FNULL:
    subprocess.check_call(['luna', list_path, '-o', db_path, '-s',
        'MASK ifnot=%s & RE & SPINDLES sig=%s max=10 fc=%g cycles=%d so mag=1.5'%(stage, ','.join(channels), cfreq, cycles)],)
    #    stdout=FNULL, stderr=subprocess.STDOUT)
    subprocess.check_call(['Rscript', r_code_path])#, stdout=FNULL, stderr=subprocess.STDOUT)
    
    df = pd.read_excel(xls_path)
    
    # delete intermediate files
    if os.path.exists(xls_path):
        os.remove(xls_path)
    if os.path.exists(r_code_path):
        os.remove(r_code_path)
    #if os.path.exists(db_path):
    #    os.remove(db_path)
        
    return df
    

if __name__=='__main__':
    # # generate aligned lists of signal and label paths

    # subject_files.xlsx should have signal_path, label_path
    # optional: MRN (or subject ID), study_date
    df = pd.read_excel('subject_files.xlsx')
    
    df = df[df.MRN!='X'].reset_index(drop=True)
    df['MRN'] = df.MRN.str.replace('-','').str.replace('/','').astype(int)
    df['study_date'] = pd.to_datetime(df.study_date)

    # # generate edf_paths and xml_paths

    output_dir = '/scr/ElissaYe/spindle_output'  # must be an absolute path
    xml_edf_dir = '/scr/ElissaYe/edf_xml'
    symlink_path = '/scr/ElissaYe/bad_files_name_replaced_by_symlink'

    channels = ['C3M2', 'C4M1', 'F3M2', 'F4M1', 'O1M2', 'O2M1']

    # generate edf and xmls files and paths required by Luna
    mrns = []
    dovs = []
    edf_paths = []
    xml_paths = []
    err_msgs = []
    for i in tqdm(range(len(df))):
        signal_path = df.signal_path.iloc[i]
        stage_path  = df.label_path.iloc[i]
        mrn = df.MRN.iloc[i]
        dov = df.study_date.iloc[i]

        edf_path = os.path.join(xml_edf_dir, os.path.basename(signal_path.replace('.mat', '.edf')))
        xml_path = os.path.join(xml_edf_dir, os.path.basename(stage_path.replace('.mat', '.xml')))
        
        try:
            if not os.path.exists(edf_path) or not os.path.exists(xml_path):
                signal_len, Fs, starttime = convert_to_edf(signal_path, edf_path, channels=channels)
                convert_to_xml(stage_path, xml_path, signal_len, Fs, starttime)
        except Exception as ee:
            # if anything goes wrong, ignore this file
            msg = '[%s]: %s'%(signal_path, str(ee))
            err_msgs.append(msg)
            print(msg)
            continue

        if os.path.exists(edf_path) and os.path.exists(xml_path):
            edf_paths.append(edf_path)
            xml_paths.append(xml_path)
            mrns.append(mrn)
            dovs.append(dov)
          
    ################################
    # luna cannot accept file names with letters below
    # for file names with "'", create symlink and replace "'" to "_"
    ################################
    edf_name_mapping = {}
    xml_name_mapping = {}
    for i in tqdm(range(len(edf_paths))):
        if "'" in edf_paths[i] or ' ' in edf_paths[i]:
            new_path = os.path.join(symlink_path, os.path.basename(edf_paths[i]).replace("'", "_").replace(' ','_'))
            if not os.path.exists(new_path):
                os.symlink(edf_paths[i], new_path)
            edf_name_mapping[edf_paths[i]] = new_path
            edf_paths[i] = new_path
        if "'" in xml_paths[i] or ' ' in xml_paths[i]:
            new_path = os.path.join(symlink_path, os.path.basename(xml_paths[i]).replace("'", "_").replace(' ','_'))
            if not os.path.exists(new_path):
                os.symlink(xml_paths[i], new_path)
            xml_name_mapping[xml_paths[i]] = new_path
            xml_paths[i] = new_path
    pd.DataFrame(data=[[os.path.basename(x).replace('.edf',''), os.path.basename(y).replace('.edf','')] for x,y in edf_name_mapping.items()], columns=['original','new']).to_excel('signal_original2new_mapping.xlsx', index=False)
    pd.DataFrame(data=[[os.path.basename(x).replace('.xml',''), os.path.basename(y).replace('.xml','')] for x,y in xml_name_mapping.items()], columns=['original','new']).to_excel('labels_original2new_mapping.xlsx', index=False)
    pd.DataFrame(data={ 'MRN':mrns, 'study_date':dovs, 'edf_path':edf_paths, 'xml_path':xml_paths}).to_csv('MRN_studydate_edf_xml_paths.csv', index=False)

    edf_paths = np.array(edf_paths)
    xml_paths = np.array(xml_paths)
    print(len(edf_paths))
    print(len(xml_paths))


    # # this is the main code to run luna


    cfreq = 13.5  # [Hz]
    cycles = 12
    stage = 'NREM2'

    # detect spindle and spindle-slow oscillation coupling

    # to make it easier to monitor the progress,
    # we separate the task into batches
    n_batch = 100
    batches = np.array_split(np.arange(len(edf_paths)), n_batch)
    dfs = []
    i_batch = 0
    for bi, batch_id in enumerate(tqdm(batches)):
        noproblem = True
        try:
            df = run_luna(output_dir,
                          [edf_paths[x] for x in batch_id],
                          [xml_paths[x] for x in batch_id],
                          channels, stage,
                          cfreq=cfreq, cycles=cycles)
        except Exception as ee:
            print('ERROR ', edf_paths[batch_id[0]], str(ee))
            noproblem = False

        if noproblem:
            dfs.append(df)

        if bi %n_batch==n_batch-1 or bi==len(batches)-1:
            df_path = os.path.join(output_dir, 'luna_output_%s_batch%d.xlsx'%(stage,i_batch+base_batch_id))
            #if os.path.exists(df_path):
            #    continue
            dfs = pd.concat(dfs, axis=0).reset_index(drop=True)
            dfs.to_excel(df_path, index=False)
            dfs = []
            i_batch += 1
            
