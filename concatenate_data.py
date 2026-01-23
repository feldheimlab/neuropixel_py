#!/usr/bin/env python3
'''
Python script to take concatenate .bin data for spike sorting through kilosort

Authors: Brian R. Mullen
Date: 2024-09-09

 
Examples:

1. Concatenate all subfolders found in the data directory
python concatenate_data.py -i D:/Main/Data/File 

2. Concatenate only the datasets indicated that are found in the data directory
python concatenate_data.py -i D:/Main/Data/File -d 0 2 3 

Saves concatenated data, ttl times, and data separation information
'''

import os 
import sys
import shutil

import numpy as np
import pandas as pd

import scipy.fftpack
from scipy.signal import butter, lfilter

import matplotlib.pyplot as plt

#change the location of this repository if needed
sys.path.append('../auditoryAnalysis/python/')
sys.path.append('../load_intan_rhd_format/')

from preprocessing import ttl_rise
try:
    from load_intan_rhd_format import *
except Exception as e:
    print(e)
    print('Cannot work with Intan files.')

from preprocessing import ttl_rise
from convert_kilosort_to_vision import get_IDs


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def get_file_org(dataset_dir: str, 
                 datasets: list,
                 intan:bool):

    print('Concatenating data ', dataset_dir)
    folders = os.listdir(dataset_dir)

    maxlength = 8

    dtype = np.dtype('int16')  # for spikeglx recordings
    nchannels = 385 # spikeglx recordings from 1.0 and 2.0

    folders_org = []
    j = 1
    first = True
    for f, folder in enumerate(folders):
        if len(folder) < maxlength:
            if not folder.endswith('.txt'):
                if datasets == None:
                    print('\tFile ', f, ':' , folder)
                    if first:
                        savefolder = folder
                        first = False
                    else:
                        savefolder += folder[-3:]
                    if intan:
                        orgdir = os.listdir(os.path.join(dataset_dir, folder))
                        org = []
                        for file in orgdir:
                            if file.endswith('.rhd'):
                                org.append(os.path.join(folder, file))
                    else:
                        org = folder + '/' + folder +'_imec0' + '/' + folder + '_t0.imec0.ap.bin'
                    folders_org.append(org)
                    # fname = os.path.join(dataset_dir, org)
                    # print('\t'+fname)
                else:
                    if int(folder[-1]) in datasets:
                        print('\tFile ', f, ':' , folder)
                        if j == 1:
                            savefolder = folder
                        else:
                            savefolder += folder[-3:]
                        if intan:
                            orgdir = os.listdir(os.path.join(dataset_dir, folder))
                            org = []
                            for file in orgdir:
                                if file.endswith('.rhd'):
                                    org.append(os.path.join(folder, file))
                        else:
                            org = folder + '/' + folder +'_imec0' + '/' + folder + '_t0.imec0.ap.bin'
                        folders_org.append(org)
                        # print('\t'+fname)
                        j += 1
    if intan:
        savef = savefolder + '/' + savefolder + '_intan-data.bin'
        print(savef)
        savefile = os.path.join(dataset_dir, savef)
        if not os.path.exists(os.path.join(dataset_dir, savefolder)):
            print('Making concatenated directory: ', os.path.join(dataset_dir, savefolder))
            os.mkdir(os.path.join(dataset_dir, savefolder))
        savefile = os.path.join(dataset_dir, savefolder) +'/' + savefolder + '_intan-data.bin'
    else:
        if not os.path.exists(os.path.join(dataset_dir, savefolder)):
            print('Making concatenated directory: ', os.path.join(dataset_dir, savefolder))
            os.mkdir(os.path.join(dataset_dir, savefolder))
        savefile = os.path.join(dataset_dir, savefolder) +'/' + savefolder + '.imec0.ap.bin'

    return folders_org, savefile


def concatentate_npx_data(dataset_dir: str,
                          folders_org: list, 
                          savefile: str):
    '''
    Conatenates data based on neuropixel recording

    Arguments:
        dataset_dir: Main directory that holds the subsets of data
        savefile: save path of concatentated data, used to get the save directory
        folders_org: all subfolders in the main dataset to include in concatenation

    Returns:
        folders_org: all subfolders in the main dataset to include in concatenation
        savefile: save path of concatentated data

    Saves:
        concatenated data in binary format, similar to how it was initially recorded 
    '''

    print('Saving to: ', savefile)
    #create bin save file
    fsave = open(savefile, 'wb')

    #open each individiual datafile to copy to the concatnetated bin file
    for datafile in folders_org:
        fo=open(os.path.join(dataset_dir, datafile), 'rb')
        shutil.copyfileobj(fo, fsave)
        fo.close()
    fsave.close()


def concatentate_intan_data(dataset_dir: str,
                            folders_org: list, 
                            savefile: str, 
                            fps: int):
    '''
    Conatenates data based on neuropixel recording

    Arguments:
        dataset_dir: Main directory that holds the subsets of data
        savefile: save path of concatentated data, used to get the save directory
        folders_org: all subfolders in the main dataset to include in concatenation

    Returns:
        folders_org: all subfolders in the main dataset to include in concatenation
        savefile: save path of concatentated data

    Saves:
        concatenated data in binary format, similar to how it was initially recorded 
    '''

    print('Saving to: ', savefile)
    #create bin save file
    datasep = [0]
    with open(savefile, 'wb') as file:
        segmentlength = 0
        for dataset in folders_org:
            # datafiles = os.listdir(os.path.join(dataset_dir, os.path.dirname(dataset)))
            # for datafile in datafiles:
            if dataset.endswith('.rhd'):
                data = read_data(os.path.join(dataset_dir, dataset))
                
                file.write(data['amplifier_data'].astype(np.int16))
                digital_data = np.array(data['board_dig_in_data'][0]).astype(np.int16)
                if segmentlength == 0:
                    timestamp = time.ctime(os.path.getctime(os.path.join(dataset_dir, dataset)))
                    print(timestamp)
                    print(data['board_dig_in_data'][0].shape)
                    
                    rised = list(ttl_rise(digital_data, fps)+segmentlength*(1000/fps))
                else:
                    rised.extend(list(ttl_rise(digital_data, fps)+(datasep[-2]+segmentlength)*(1000/fps)))
                segmentlength += data['amplifier_data'].shape[1]*(1000/fps)
            datasep.append(datasep[-1] + segmentlength)
    close(savefile)
    savedir = os.path.dirname(savefile)

    np.save(os.path.join(savedir, 'ttlTimes.npy'), rised)
    np.save(os.path.join(savedir, 'datasep.npy'), {'Datasep':datasep, 'Datalength':segmentlength, 'Timestamp':timestamp})


def ttl_npx_data(dataset_dir: str, 
                 savefile: str, 
                 folders_org: list, 
                 fps: int):
    '''
    Uses memory maps to load only the digital data (last channel)
    
    Arguments:
        dataset_dir: Main directory that holds the subsets of data
        savefile: save path of concatentated data, used to get the save directory
        folders_org: all subfolders in the main dataset to include in concatenation
    
    Returns:
        None

    Saves two files:
        ttl times: numpy file saving all times TTLs occured, relative the concatenetated data 
        Datasep: Dictionary containing data separation times and dataset lengths
    '''

    savedir = os.path.dirname(savefile)

    print('\nCalculating TTLs and data separation.\n')
    
    # Hardcoded based on neuropixel data
    dtype = np.dtype('int16')  # for spikeglx recordings 
    nchannels = 385 # spikeglx recordings from 1.0 and 2.0
    dig_channel = 384
    overlap = 5

    # Variables to save
    total_time = 0
    segmentlength = []
    datasep = [0]
    ttls = []
    first = True
    for datafile in folders_org:
        fname = os.path.join(dataset_dir, datafile)
        print('Working on: ', fname)
        if first:
            timestamp = time.ctime(os.path.getctime(fname))
            first = False
        # calculate the sample size from the filesize
        nsamples = os.path.getsize(fname)/(nchannels*dtype.itemsize)
        segmentlength.append(1000*nsamples/fps) # in ms
        total_time += segmentlength[-1]
        datasep.append(total_time)

        #set up memory map
        dat = np.memmap(fname,
                mode='r', # open in read mode (safe)
                dtype=dtype,
                shape = (int(nsamples),int(nchannels)))

        #read the data in batches
        batchsz = 60 * fps #batch size
        batches = np.arange(batchsz,int(nsamples)+batchsz, batchsz)
        print('\tTotal time of dataset: {} sec'.format(np.round(segmentlength[-1]/1000)))
        print('\tTotal number of batches (1 minute each): ', len(batches))
        
        for b, batch in enumerate(batches):
            if (b % 10)==0:
                print('\t\tWorking on batch :', b)
            if b == 0:
                digital = dat[:batch+overlap, dig_channel].astype('float16')
                digital[digital>0.5]=1
                ttls.extend(list(ttl_rise(digital, rate=fps)+datasep[-2]))
            else:
                digital = dat[batches[b-1]:batch+overlap, dig_channel].astype('float16')
                digital[digital>0.5]=1
                ttls.extend(list(ttl_rise(digital, rate=fps)+datasep[-2]+batches[b-1]*(1000/fps)))

    print('\nDatasep:', datasep)
    print('Datalength:', segmentlength)
    print('Total TTLs found:', len(ttls))

    print('Saving TTL and Data Seperation data: ', savedir)
    np.save(os.path.join(savedir, 'ttlTimes.npy'), ttls)
    np.save(os.path.join(savedir, 'datasep.npy'), {'Datasep':datasep, 'Datalength':segmentlength, 'Timestamp':timestamp})


def welford_stat_update(count, mean, M2, newdata):
    count += 1
    delta = newdata - mean
    mean += delta / count
    delta2 = newdata - mean
    M2 += delta * delta2
    
    return count, mean, M2


def welford_stat_finalize(count, mean, M2):
    if count < 2:
        return mean, np.nan
    else:
        variance = M2 / count
        return mean, variance


def make_waveform_summary(dataset_dir: str,
                 kilosortloc: str,  
                 savefile: str, 
                 folders_org: list,
                 spiketemplates: list,
                 spiketimes: list,
                 cluster_ids: list,
                 cluster_index: list, 
                 fps: int = 30000):
    '''
    Uses memory maps to load only the digital data (last channel)
    
    Arguments:
        dataset_dir: Main directory that holds the subsets of data
        savefile: save path of concatentated data, used to get the save directory
        folders_org: all subfolders in the main dataset to include in concatenation
    
    Returns:
        None

    Saves two files:
        ttl times: numpy file saving all times TTLs occured, relative the concatenetated data 
        Datasep: Dictionary containing data separation times and dataset lengths
    '''
    print('\nCalculating waveforms.\n')
    
    # Hardcoded based on neuropixel data
    dtype = np.dtype('int16')  # for spikeglx recordings 
    nchannels = 385 # spikeglx recordings from 1.0 and 2.0
    dig_channel = 384

    # Variables to save
    total_time = 0
    datalength = []
    datasep = [0]
    
    waveforms = np.zeros((len(cluster_ids), 61, nchannels-1))
    waveforms_var = np.zeros_like(waveforms)
    nwaveform = np.zeros(len(cluster_ids))
    nwaveform_act = np.zeros(len(cluster_ids))

    for dataf, datafile in enumerate(folders_org):
        fname = os.path.join(dataset_dir, datafile)
        print('Working on: ', fname)
        # calculate the sample size from the filesize
        nsamples = os.path.getsize(fname)/(nchannels*dtype.itemsize)
        datalength.append(1000*nsamples/fps) # in ms
        total_time += datalength[-1]
        datasep.append(total_time)

        #set up memory map
        dat = np.memmap(fname,
                mode='r', # open in read mode (safe)
                dtype=dtype,
                shape = (int(nsamples),int(nchannels)))

        #read the data in batches
        batchsz = 60 * fps #batch size
        batches = np.arange(batchsz,int(nsamples)+batchsz, batchsz)
        print('\tTotal time of dataset: {} sec'.format(np.round(datalength[-1]/1000)))
        print('\tTotal number of batches (1 minute each): ', len(batches))
        
        for b, batch in enumerate(batches):
            if (b % 10)==0:
                print('\t\tWorking on batch :', b)
            if b == 0:
                stimes = np.array(spiketimes[spiketimes<batch])
                stemps = spiketemplates[spiketimes<batch]
                data = dat[:batch, :].astype('float16')
            else:
                stimes = np.array(spiketimes[(spiketimes<batch)&(spiketimes>batches[b-1])] - batches[b-1])
                stemps = spiketemplates[(spiketimes<batch)&(spiketimes>batches[b-1])]
                data = dat[batches[b-1]:batch, :].astype('float16')
            
            for d in np.arange(nchannels): #apply filter across all channels
                data[:,d] -= data[:,d].mean()
                data[:,d] = butter_bandpass_filter(data[:,d], lowcut = 1000, highcut = (fps-100)/2, fs=fps)
            
            waves = np.unique(stemps)
            matrix_size = data.shape[0]
            for w in waves:
                ind = cluster_index[w==cluster_ids]    
                ctime = stimes[stemps==w]
                nwaveform_act[ind]+=ctime.shape[0]
                for t in ctime:
                    if (t>20)&(t<(matrix_size-41)):
                        try:
                            nwaveform[ind], waveforms[ind,:,:], waveforms_var[ind,:,:] = welford_stat_update(nwaveform[ind], waveforms[ind,:,:], waveforms_var[ind,:,:], data[int(t-20):int(t+41),:-1])
                        except Exception as e:
                            print('Missing updated metrics, due to ', data[int(t-20):int(t+41),:-1].shape, )
    for w in waves:
        ind = cluster_index[w==cluster_ids]                       
        waveforms[ind,:,:], waveforms_var[ind,:,:] = welford_stat_finalize(nwaveform[ind], waveforms[ind,:,:], waveforms_var[ind,:,:])                    
        
    print('Saving updated templates data: ', kilosortloc)  
    np.save(os.path.join(kilosortloc, 'templates_mean.npy'), waveforms)
    np.save(os.path.join(kilosortloc, 'templates_vars.npy'), waveforms_var)
    np.save(os.path.join(kilosortloc, 'nwaveform.npy'), nwaveform)


def fft_raw_data(dataset_dir:str,
                 folders_org:str,
                 savefile:str,
                 fps: int,
                 time_offset:float=30,
                 time_dur:float=60):

    dtype = np.dtype('int16')  # for spikeglx recordings 
    nchannels = 385 # spikeglx recordings from 1.0 and 2.0

    for datafile in folders_org:
        fname = os.path.join(dataset_dir, datafile)

        # calculate the sample size from the filesize
        nsamples = os.path.getsize(fname)/(nchannels*dtype.itemsize)
        dat = np.memmap(fname,
                        mode='r', # open in read mode (safe)
                        dtype=dtype,
                        shape = (int(nsamples),int(nchannels)))

        # lets take the samples from around 30sec into the recording
        idx = np.arange(fps*time_dur).astype(int)
        print('FFT analysis: {} seconds used for analysis'.format(idx.shape[0]/fps))
        nplot = 10
        diff = 0
        print('\tOnly {} timeseries from various locations on the probe are plotted'.format(nplot))
        fig, axs = plt.subplots(2,1, figsize=(15,5))
        plotchanels = nchannels//nplot
        for ichan in np.arange(nchannels):
            y = dat[idx+int(time_offset*fps),ichan].astype('float32')
            #subtract the mean (for plotting)
            y -= y.mean()
            if ichan%plotchanels==0:
                ploty = y[:fps//2]-np.convolve(y[:fps//2], np.ones(100)/100, mode='same')                
                axs[0].plot(idx[:fps//2]/fps, ploty+diff, color='k', linewidth=1, alpha=1)
                axs[0].text(0.5, diff, ichan)
                diff += np.abs(np.min(ploty))*3
            yf = scipy.fftpack.fft(y)
            if ichan==0:
                T = 1.0 / fps
                N = idx.shape[0]
                xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
            axs[1].plot(xf, np.convolve(2.0/N * np.abs(yf[:N//2]), np.ones(10), mode='same'), color='k', alpha=1/255)

        axs[0].set_xlabel('time (s)')
        axs[0].set_ylabel('rel. amp')
        axs[1].set_xlim(0,1000)
        axs[1].set_xticks(np.arange(0,1000,60))
        axs[1].set_ylim(.01,400)
        axs[1].set_xlabel('frequency')
        axs[1].set_ylabel('power')
        axs[1].set_yscale('log')
        plt.tight_layout()
        parentdir = os.path.dirname(os.path.dirname(fname))
        savename = os.path.join(os.path.dirname(savefile), '{}_fft.png'.format(os.path.basename(parentdir)))
        print('\tSaving fft data: ', savename)  
        plt.savefig(savename, dpi=300)


if __name__ == '__main__':

    import argparse
    import time
    import datetime

    # Argument Parsing
    # -----------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input_directory', type = str,
        required = True, 
        help = 'path to the .bin files for concatenation')
    ap.add_argument('-d', '--datasets', type = list, 
        nargs = '+', required = False, default = None,
        help = 'list of datasets to include, if left blank all datasets in the input directory will be included')
    ap.add_argument('-con', '--concatenate', action='store_true',
        help='Boolean indicating it will concatenate the data anew')
    ap.add_argument('-w', '--waveform', action='store_true',
        help='Boolean indicating it will create the summary of the waveforms. This will take a long time.')
    ap.add_argument('-fft', '--fft', action='store_true',
        help='Boolean indicating it will run the FFT on the raw data.')
    ap.add_argument('-in', '--intan', action='store_true',
        help='Boolean indicating it will assume intan organizated data.')
    ap.add_argument('-g', '--group_tsv', type = str,
        default='cluster_info.tsv',  
        help = 'tsv with specified classified clusters')
    ap.add_argument('-c', '--class_col', type = str,
        default='KSLabel', 
        help = 'column name of group tsv that indicates that classification of neuronal dataset')
    ap.add_argument('-f', '--fps', type = int,
        default=30000,  
        help = 'frames per second for data collection')
    args = vars(ap.parse_args())

    dataset_dir = args['input_directory']
    waveform = args['waveform']
    concatenate = args['concatenate']
    intan = args['intan']
    fft = args['fft']
    fps = args['fps']
    if intan:
        fps = 20000
    clusterdef = args['group_tsv']
    class_col = args['class_col']
    matlab_version = '5'

    assert os.path.exists(dataset_dir), 'Data directory does not exist: {}'.format(dataset_dir)
    
    if args['datasets'] != None:
        datasets = args['datasets']
        for d, data in enumerate(datasets):
            if len(data)==1:
                datasets[d] = int(np.squeeze(data))
            else:
                number = ''
                for da in data:
                    number += da
                datasets[d] = int(number)
    else:
        datasets = None

    if datasets!=None:
        if len(datasets) == 1:
            print('Only one datafile specified.')
            folders = os.listdir(dataset_dir)
            for f, folder in enumerate(folders):
                if not folder.endswith('.txt'):
                    if int(folder[-1]) in datasets:
                        print('\tFile ', f, ':' , folder)
                        if intan:
                            orgdir = os.listdir(os.path.join(dataset_dir, folder))
                            folders_org = []
                            for file in orgdir:
                                if file.endswith('.rhd'):
                                    folders_org.append(os.path.join(folder, file))
                            savef = folder + '/' + folder + '_intan-data.bin'
                            savefile = os.path.join(dataset_dir, savef)
                        else:
                            print('Skipping concatenation step.')
                            folders_org = folder + '/' + folder +'_imec0' + '/' + folder + '_t0.imec0.ap.bin'
                            savefile = os.path.join(dataset_dir, folders_org)
                            assert os.path.exists(savefile), 'Datafile does not exist: {}'.format(savefile)
                            print('Found datafile: '+ savefile)
            # if intan:
            #     concatentate_intan_data(dataset_dir, folders_org, savefile, fps)
            # else:
            #     ttl_npx_data(dataset_dir, savefile, folders_org, fps)
        # else:
        #     folders_org, savefile = get_file_org(dataset_dir, datasets, intan)
            # if intan:
            #    concatentate_intan_data(dataset_dir, folders_org, savefile, fps) 
            # else:
            #     if concatenate:
            #         concatentate_npx_data(dataset_dir, folders_org, savefile)
            #     ttl_npx_data(dataset_dir, savefile, folders_org, fps)
    # else:
    #     folders_org, savefile = get_file_org(dataset_dir, datasets, intan)
        # if intan:
        #     concatentate_intan_data(dataset_dir, folders_org, savefile, fps) 
        # else:
        #     if concatenate:
        #         concatentate_npx_data(dataset_dir, folders_org, savefile)
        #     ttl_npx_data(dataset_dir, savefile, folders_org, fps)
    
    if fft:
        fft_raw_data(dataset_dir, folders_org, savefile, fps)

    if waveform:
        savefile = dataset_dir
        print(savefile)
        folders_org = [os.path.join(dataset_dir, 'data_g0_tcat.imec0.ap.bin')]
        try:
            kilosortloc = os.path.join(savefile, 'kilosort4')
            assert os.path.exists(kilosortloc), 'Could not find: {}'.format(kilosortloc)
        except:
            kilosortloc = os.path.join(savefile, 'kilosort3')
        assert os.path.exists(kilosortloc), 'Could not find kilosort files: {}\
            \nThese files should be located in the same directory as the concatenated data.'.format(kilosortloc)

        #spikes information
        cluster = pd.read_csv(os.path.join(kilosortloc, clusterdef), sep='\t') #class 
        cluster_ids, cluster_index = get_IDs(cluster, class_col='KSLabel', matlab_version=matlab_version, group='all')
        #spike cluster in order of event
        spiketemplates = np.load(os.path.join(kilosortloc, 'spike_clusters.npy')) # to make asdf
        #event times
        spiketimes = np.load(os.path.join(kilosortloc, 'spike_times.npy')) # to make asdf
        #waveforms for original clusters
        # templates = np.load(os.path.join(kilosortloc, 'templates.npy')) #waveforms

        # orig_templateloc = os.path.join(kilosortloc, 'templates_orig.npy')
        # if os.path.exists(orig_templateloc):
        #     print('Skipping saving orginal templates, as this has already been done.\n\t', orig_templateloc)
        # else:
        #     print('Saving orginal templates:\n\t', orig_templateloc)
        #     np.save(orig_templateloc, templates) #waveforms
        
        IDs, IDs_index = get_IDs(cluster, class_col, matlab_version=matlab_version, group='good')

        make_waveform_summary(dataset_dir, kilosortloc, savefile, folders_org, spiketemplates, spiketimes,
                              cluster_ids, cluster_index, fps)