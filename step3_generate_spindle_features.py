import os
import subprocess
import numpy as np
import pandas as pd
import pyedflib
from tqdm import tqdm
from load_dataset import *


def convert_to_edf(input_path, output_path, dataset):
    """
    """
    _load_dataset = eval(f'load_{dataset}_dataset')
    EEG, sleep_stages, EEG_channels, combined_EEG_channels, Fs, start_time = _load_dataset(signal_path, annot_path)
            
    channel_info = [
            {'label': EEG_channels[i],
             'dimension': 'uV',
             'sample_rate': Fs,
             'physical_max': 32767,
             'physical_min': -32768,
             'digital_max': 32767,
             'digital_min': -32768,
             'transducer': 'E',
             'prefilter': ''}
        for i in range(len(EEG_channels))]

    with pyedflib.EdfWriter(output_path, len(EEG), file_type=pyedflib.FILETYPE_EDFPLUS) as ff:
        ff.setSignalHeaders(channel_info)
        ff.writeSamples(EEG)
            
    return EEG.shape[1], Fs, start_time, sleep_stages


def convert_to_xml(sleep_stages, output_path, Fs, epoch_sec=30):
    """
    """       
    sleep_stages = sleep_stages[np.arange(0, len(sleep_stages), int(round(Fs*epoch_sec)))]
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


def run_luna(output_dir, sids, edf_paths, xml_paths, channels, stage, cfreq=13.5, cycles=12):
    # create the list file
    list_path = os.path.join(output_dir, 'luna.lst')
    df = pd.DataFrame(data={'sid': sids, 'edf':edf_paths, 'xml':xml_paths})
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
    # get list of files
    df_mastersheet = pd.read_excel('mastersheet.xlsx')
    
    # luna does not allow space in sid
    df_mastersheet['SID2'] = df_mastersheet.SID.str.replace(' ','_')
    
    # # generate edf_paths and xml_paths
    #TODO define
    # this is usually not a dropbox folder since this will be big
    # must be an absolute path without space!
    output_dir = '/data/dropbox_dementia_detection_ElissaYe_spindle_results'
    
    output_feature_dir = 'features'
    if not os.path.exists(output_feature_dir): os.mkdir(output_feature_dir)
    # folder that contains the input edf and xml files
    xml_edf_dir = os.path.join(output_dir, 'edf_xml')
    if not os.path.exists(xml_edf_dir): os.mkdir(xml_edf_dir)

    # generate edf and xmls files and paths required by Luna
    edf_paths = []
    xml_paths = []
    err_msgs = []
    for i in tqdm(range(len(df_mastersheet))):
        sid = df_mastersheet.SID2.iloc[i]
        signal_path = df_mastersheet.SignalPath.iloc[i]
        annot_path  = df_mastersheet.AnnotPath.iloc[i]
        dataset = df_mastersheet.Dataset.iloc[i]

        edf_path = os.path.join(xml_edf_dir, sid+'.edf')
        xml_path = os.path.join(xml_edf_dir, sid+'.xml')
        
        try:
            if not os.path.exists(edf_path) or not os.path.exists(xml_path):
                signal_len, Fs, start_time, annot = convert_to_edf(signal_path, edf_path, dataset)
                convert_to_xml(annot, xml_path, Fs)
        except Exception as ee:
            # if anything goes wrong, ignore this file
            msg = '[%s]: %s'%(signal_path, str(ee))
            err_msgs.append(msg)
            print(msg)
            continue

        if os.path.exists(edf_path) and os.path.exists(xml_path):
            edf_paths.append(edf_path)
            xml_paths.append(xml_path)

    edf_paths = np.array(edf_paths)
    xml_paths = np.array(xml_paths)
    print(len(edf_paths))
    print(len(xml_paths))


    # # this is the main code to run luna

    cfreq = 13.5  # [Hz]
    cycles = 12
    stage = 'N2'

    # detect spindle and spindle-slow oscillation coupling
    
    dfs = []
    for i in tqdm(range(len(df_mastersheet))):
        noproblem = True
        dataset = df_mastersheet.Dataset.iloc[i]
        _get_channels = eval(f'get_{dataset}_channels')
        ch_names, standard_ch_names, pair_ch_ids, combined_ch_names = _get_channels()
        try:
            df_res = run_luna(
                output_dir,
                df_mastersheet.SID2[[i]].values,
                [edf_paths[i]],
                [xml_paths[i]],
                ch_names, stage,
                cfreq=cfreq, cycles=cycles)
            df_res.insert(list(df_res.columns).index('ID')+1, 'Dataset', [dataset]*len(df_res))
        except Exception as ee:
            print('ERROR ', str(ee))
            noproblem = False

        if noproblem:
            dfs.append(df_res)
            
    dfs = pd.concat(dfs, axis=0, ignore_index=True)
    dfs.to_csv(os.path.join(output_feature_dir, f'luna_output_{stage}.csv'), index=False)
    
    
    # convert to one subject per row
    datasets = dfs.Dataset.unique()
    for dataset in datasets:
        _get_channels = eval(f'get_{dataset}_channels')
        ch_names, standard_ch_names, pair_ch_ids, combined_ch_names = _get_channels()
        
        df = dfs[dfs.Dataset==dataset].reset_index(drop=True)
        
        sids = sorted(set(df.ID))
        vals = []
        for sid in sids:
            val = []
            for ids in pair_ch_ids:
                val.append(np.mean(np.r_[
                    df[(df.ID==sid)&(df.CH==ch_names[ids[0]])].iloc[:,5:].values,
                    df[(df.ID==sid)&(df.CH==ch_names[ids[1]])].iloc[:,5:].values,
                ], axis=0))
            vals.append( np.concatenate(val) )
    
        df2 = pd.DataFrame(
            data=np.array(vals), columns=np.concatenate([[x+'_'+ch for x in df.columns[5:]] for ch in combined_ch_names]))
        df2['SID'] = sids
        
        df3 = df_mastersheet[['SID', 'Dataset']].merge(df2, on='SID', how='inner')
        df3.to_csv(os.path.join(output_feature_dir, f'spindle_features_{stage}_channel_avg_{dataset}.csv'), index=False)

