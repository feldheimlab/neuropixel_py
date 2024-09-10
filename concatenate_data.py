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

#change the location of this repository if needed
sys.path.append('../auditoryAnalysis/python/')
from preprocessing import ttl_rise


def concatentate_npx_data(dataset_dir: str, 
                          datasets: list):
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
                ttls.extend(list(ttl_rise(digital, rate=fps)+datasep[-2]+b*batchsz))
            else:
                digital = dat[batches[b-1]:batch, dig_channel].astype('float16')
                digital[digital>0]=1
                ttls.extend(list(ttl_rise(digital, rate=fps)+datasep[-2]+b*batchsz))

    # for t, ttl in enumerate(ttls):
    #     if (ttl - ttls[t-1])<10:
    #         ttls.remove(ttl)

    print('\nDatasep:', datasep)
    print('Datalength:', datalength)
    print('Total TTLs found:', len(ttls))

    print('Saving TTL and Data Seperation data: ', savedir)
    np.save(os.path.join(savedir, 'TTLs.npy'), ttls)
    np.save(os.path.join(savedir, 'datasep.npy'), {'Datasep':datasep, 'Datalength':datalength})


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
    args = vars(ap.parse_args())

    dataset_dir = args['input_directory']

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
                    org = folder + '/' + folder +'_imec0' + '/' + folder + '_t0.imec0.ap.bin'
                    datafile = os.path.join(dataset_dir, org)
                    assert os.path.exists(datafile), 'Datafile does not exist: {}'.format(datafile)
                    print('Found datafile: '+ datafile)

            ttl_npx_data(dataset_dir, datafile, folders_org=org)
        else:
            folders_org, savefile = concatentate_npx_data(dataset_dir, datasets)
            ttl_npx_data(dataset_dir, savefile, folders_org)
    else:
        folders_org, savefile = concatentate_npx_data(dataset_dir, datasets)
        ttl_npx_data(dataset_dir, savefile, folders_org)


