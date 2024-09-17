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
import scipy.fftpack
from scipy.signal import butter, lfilter

import matplotlib.pyplot as plt

#change the location of this repository if needed
sys.path.append('../auditoryAnalysis/python/')
from preprocessing import ttl_rise
from convert_kilosort_to_vision import get_IDs


def concatentate_npx_data(dataset_dir: str, 
                          datasets: list,
                          concatenate:bool):
    '''
    Conatenates data based on neuropixel recording

    Arguments:
        dataset_dir: Main directory that holds the subsets of data
        datasets: list of data subsets to be included in the concatentated data

    Returns:
        folders_org: all subfolders in the main dataset to include in concatenation
        savefile: save path of concatentated data

    Saves:
        concatenated data in binary format, similar to how it was initially recorded 
    '''

    print('Concatenating data ', dataset_dir)
    folders = os.listdir(dataset_dir)[:2]

    dtype = np.dtype('int16')  # for spikeglx recordings
    nchannels = 385 # spikeglx recordings from 1.0 and 2.0

    folders_org = []
    j = 1

    for f, folder in enumerate(folders):
        if datasets == None:
            print('\tFile ', f, ':' , folder)
            if f == 0:
                savefolder = folder
            else:
                savefolder += folder[-3:]
            org = folder + '/' + folder +'_imec0' + '/' + folder + '_t0.imec0.ap.bin'
            folders_org.append(org)
            fname = os.path.join(dataset_dir, org)
            # print('\t'+fname)
        else:
            if int(folder[-1]) in datasets:
                print('\tFile ', f, ':' , folder)
                if j == 1:
                    savefolder = folder
                else:
                    savefolder += folder[-3:]
                org = folder + '/' + folder +'_imec0' + '/' + folder + '_t0.imec0.ap.bin'
                folders_org.append(org)
                fname = os.path.join(dataset_dir, org)
                # print('\t'+fname)
                j += 1

    if not os.path.exists(os.path.join(dataset_dir, savefolder)):
        print('Making concatenated directory: ', os.path.join(dataset_dir, savefolder))
        os.mkdir(os.path.join(wd, savefolder))
    savefile = os.path.join(dataset_dir, savefolder) +'/' + savefolder + '.imec0.ap.bin'

    if concatenate:
        print('Saving to: ', savefile)
        #create bin save file
        fsave = open(savefile, 'wb')

        #open each individiual datafile to copy to the concatnetated bin file
        for datafile in folders_org: 
            fo=open(os.path.join(dataset_dir, datafile), 'rb')
            shutil.copyfileobj(fo, fsave)
            datasep.append()
            fo.close()
        fsave.close()

    return folders_org, savefile


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

    # Variables to save
    total_time = 0
    datalength = []
    datasep = [0]
    ttls = []

    for datafile in folders_org:
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
                digital = dat[:batch, dig_channel].astype('float16')
                digital[digital>0]=1
                ttls.extend(list(ttl_rise(digital, rate=fps)+datasep[-1]))
            else:
                digital = dat[batches[b-1]:batch, dig_channel].astype('float16')
                digital[digital>0]=1
                ttls.extend(list(ttl_rise(digital, rate=fps)+datasep[-1]+batches[b-1]*(1000/fps)))
 

    ttls = np.array(ttls)
    # for t, ttl in enumerate(ttls):
    #     if (ttl - ttls[t-1])<10:
    #         ttls.remove(ttl)



    print('\nDatasep:', datasep)
    print('Datalength:', datalength)
    print('Total TTLs found:', len(ttls))

    print('Saving TTL and Data Seperation data: ', savedir)
    np.save(os.path.join(savedir, 'ttlTimes.npy'), ttls)
    np.save(os.path.join(savedir, 'datasep.npy'), {'Datasep':datasep, 'Datalength':datalength})


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
    waveforms = np.zeros((len(cluster_ids), nchannels, 61))#double check this matches the template file
    nwaveform = np.zeros(len(cluster_ids))

    for datafile in folders_org:
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
                stimes = np.array(spiketimes[spiketimes<batch])*fps/1000
                stemps = spiketemplates[spiketimes<batch]
                data = dat[:batch, :].astype('float16')
            else:
                stimes = np.array(spiketimes[(spiketimes<batch)&(spiketimes>batches[b-1])] - batches[b-1]*(1000/fps))*fps/1000
                stemps = spiketemplates[(spiketimes<batch)&(spiketimes>batches[b-1])]
                data = dat[batches[b-1]:batch, :].astype('float16')
            
            for d in np.arange(nchannels): #apply filter across all channels
                data[:,d] -= data[:,d].mean()
                data[:,d] = butter_bandpass_filter(data[:,d], lowcut = 1000, highcut = (fps-100)/2, fs=fps)
            
            waves = np.unique(stemps)
            for w in waves:
                ind = cluster_index[w==cluster_id]    
                ctime = stimes[stemps==w]
                for t in ctime:
                    if (t>20)&(t<(batchsz-41)):
                        waveforms[ind,:,:] += data[int(t-20):int(t+41),:]
                        nwaveform[ind] += 1

    for ind in cluster_index:                    
        waveforms[ind] =/ nwaveform[ind]

    print('Saving updated templates data: ', kilosortloc)  
    np.save(os.path.join(kilosortloc, 'templates.npy'), waveforms)


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
        print('{} seconds used for analysis'.format(idx.shape[0]/fps))
        nplot = 10
        print('Only the first {} timeseries are plotted'.fomrat(nplot))
        fig, axs = plt.subplots(2,1, figsize=(15,5))
        for ichan in np.arange(nchannels):
            y = dat[idx+int(time_offset*fps),ichan].astype('float32')
            #subtract the mean (for plotting)
            y -= y.mean()
            if ichan<nplot:                
                axs[0].plot(idx[:fps]/fps,y[:fps]+i*200, color='k', alpha=0.1)

            yf = scipy.fftpack.fft(y)
            if ichan==0:
                T = 1.0 / fps
                N = idx.shape[0]
                xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
            axs[1].plot(xf, np.convolve(2.0/N * np.abs(yf[:N//2]), np.ones(10), mode='same'), color='k', alpha=0.01)

        axs[0].set_xlabel('time (s)')
        axs[0].set_ylabel('rel. amp')
        axs[1].set_xlim(0,320)
        axs[1].set_xlabel('frequency')
        axs[1].set_ylabel('power')
        axs[1].set_yscale('log')
        plt.tight_layout()
        parentdir = os.path.dirname(fname)
        print('Saving fft data: ', os.path.dirname(savefile))  
        plt.savefig(os.path.join(os.path.dirname(savefile), '{}_fft.png'.format(parentdir[-2:]), dpi=300)


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
    ap.add_argument('-c', '--concatenate', type=bool,
        default=False,
        help='Boolean indicating it will concatenate the data anew')
    ap.add_argument('-w', '--waveform', type=bool,
        default=False,
        help='Boolean indicating it will create the summary of the waveforms. This will take a long time.')
    ap.add_argument('-fft', '--fft', type=bool,
        default=True,
        help='Boolean indicating it will run the FFT on the raw data.')
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
    clusterdef = args['group_tsv']
    class_col = args['class_col']
    matlab_version = '5'
    fps = args['fps']

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
            print('Only one datafile specified. Skipping concatenation step.')
            folders = os.listdir(dataset_dir)[:2]
            for f, folder in enumerate(folders):
                if int(folder[-1]) in datasets:
                    print('\tFile ', f, ':' , folder)
                    folders_org = folder + '/' + folder +'_imec0' + '/' + folder + '_t0.imec0.ap.bin'
                    datafile = os.path.join(dataset_dir, org)
                    assert os.path.exists(datafile), 'Datafile does not exist: {}'.format(datafile)
                    print('Found datafile: '+ datafile)

            ttl_npx_data(dataset_dir, datafile, folders_org=folders_org, fps)
        else:
            folders_org, savefile = concatentate_npx_data(dataset_dir, datasets, concatenate)
            ttl_npx_data(dataset_dir, savefile, folders_org, fps)
    else:
        folders_org, savefile = concatentate_npx_data(dataset_dir, datasets, concatenate)
        ttl_npx_data(dataset_dir, savefile, folders_org, fps)
    
    if fft:
        fft_raw_data(dataset_dir, folders_org, savefile, fps, time_offset, time_dur)

    if waveform:
        try:
            kilosortloc = os.path.join(os.path.dirname(savefile), 'kilosort4')
            assert os.path.exists(kilosortloc), 'Could not find: {}'.format(kilosortloc)
        except:
            kilosortloc = os.path.join(os.path.dirname(savefile), 'kilosort3')
        assert os.path.exists(kilosortloc), 'Could not find kilosort files.\
            \nThese files should be located in the same directory as the concatenated data.'

        #spikes information
        cluster = pd.read_csv(os.path.join(kilosortloc, clusterdef), sep='\t') #class 
        #spike cluster in order of event
        spiketemplates = np.load(os.path.join(kilosortloc, 'spike_clusters.npy')) # to make asdf
        #event times
        spiketimes = np.load(os.path.join(kilosortloc, 'spike_times.npy')) # to make asdf
        #waveforms for original clusters
        templates = np.load(os.path.join(kilosortloc, 'templates.npy')) #waveforms
        np.save(os.path.join(kilosortloc, 'templates_orig.npy'), templates) #waveforms
        
        IDs, IDs_index = get_IDs(cluster, class_col, matlab_version=matlab_version, group='good')

        make_waveform_summary(dataset_dir, kilosortloc, savefile, folders_org, spiketemplates, spiketimes,
                              cluster_ids, cluster_index, fps)


