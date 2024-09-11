#!/usr/bin/env python3
'''
Python script to take kilosort data and convert it to vision-like outputs

Authors: Brian R. Mullen
Date: 2024-09-09

'''
import os
import sys

import numpy as np

from scipy.io import savemat

def ttl_rise_d(digital_data, rate=rate):
    digital_data = np.squeeze(digital_data)
    dif = digital_data[:-1]-digital_data[1:]
    rise = np.where(dif<-0.5)[0]
    rise_d = []
    for r, ris in enumerate(rise):
        if r == 0:
            rise_d.append(ris)
        elif (ris) >= (rise_d[-1]+(.0095*rate)):
            rise_d.append(ris)
            
    return np.array(rise_d)/rate*1000


def get_IDs(cluster, group='good'):
    IDs = cluster[cluster['group']==group].index
    return np.array(IDs, dtype=int)

def ttlTimes_creation(wd, dig, rate=rate):
    print('Making ttlTimes.mat file')
    ttlTimes = ttl_rise_d(dig, rate=rate)
    savedict = {'ttlTimes':ttlTimes}

    print('\tSaving ttlTimes.mat to: ', wd)
    savemat(os.path.join(wd, 'ttlTimes.mat'), savedict, format='7.2')

def eisummary_creation(wd, spiketemplates, IDs, cluster, rate=rate):
    print('Creating eisummary.mat file.')
    waveform = spiketemplates[IDs]
    center = np.nanargmin(waveform, axis=1)
    eitime = np.arange(-center, waveform.shape[0])/(rate/1000)
    maxChannels  = cluster.loc[IDs, 'ch'].values
    maxAmplitudes = cluster.loc[IDs, 'Amplitude'].values
    savedict = {'waveform':waveform, 
                'eitime':eitime, 
                'maxChannels':maxChannels, 
                'maxAmplitudes':maxAmplitudes}

    print('\tSaving xy.mat to: ', wd)
    savemat(os.path.join(wd, 'xy.mat'), savedict, format='7.2')

def segmentlengths_creation(datasep):
    print('Creating segmentlengths.mat file')
    timestamps = 
    segmentlengths = np.diff(datasep)
    segmentseparations = datasep[1:]
    savedict = {'timestamps':timestamps, 
                'segmentlengths':segmentlengths, 
                'segmentseparations':segmentseparations}

    print('\tSaving segmentlengths.mat to: ', wd)
    savemat(os.path.join(wd, 'segmentlengths.mat'), savedict, format='7.2')

def xy_creation():
    print('Making xy.mat file')
    x = chanposition[:,0]
    y = chanposition[:,1]
    xs = chanposition[:,0]
    ys  = chanposition[:,1]
    ang = np.zeros_like(chanposition[:,0])
    savedict = {'x':x, 
                'y':y, 
                'xs':xs, 
                'ys':ys, 
                'ang':ang}

    print('\tSaving xy.mat to: ', wd)
    savemat(os.path.join(wd, 'xy.mat'), savedict, format='7.2')

def basicinfo_creation(recordingregion='SC'):
    print('Making basicinfo.mat file')
    savedict = {'recordingregion':recordingregion}
    print('\tSaving basicinfo.mat to: ', wd)
    savemat(os.path.join(wd, 'basicinfo.mat'), savedict, format='7.2')

def asdf_creation(IDs, spiketimes, spiketemplates):
    print('Making asdf.mat file')
    asdf = np.empty(IDs.shape, dtype=object)
    
    for n, neuron in enumerate(IDs):
        try:
            asdf[n]=np.array(np.squeeze(spiketimes[spiketemplates==neuron]))

    IDs
    asdf_raw
    location


def convert_kilosort_to_vision(dig, cluster, rate=rate, group='good'):
    ttlTimes_creation(dig, rate=rate)
    IDs = get_IDs(cluster, group=group)


rate = 20000

if __name__ == '__main__':

    import argparse
    import time
    import datetime

    # Argument Parsing
    # -----------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input_directory', type = str,
        required = True, 
        help = 'path to the kilosort4 files for concatenation')
    args = vars(ap.parse_args())

    kilosortloc = args['input_directory']

    #spikes
    spiketemplates = np.load(os.path.join(kilosortloc, 'spike_clusters.npy')) # to make asdf
    spiketimes = np.load(os.path.join(kilosortloc, 'spike_times.npy')) # to make asdf
    templates = np.load(os.path.join(kilosortloc, 'templates.npy')) #waveforms

    cluster = pd.read_csv(os.path.join(kilosortloc, 'cluster_info.tsv'), sep='\t') #class 

    #created from rhd files Save the ttl times in a file so we don't have to determine this every time
    digital = np.load(os.path.join(kilosortloc, 'digital.npy')) #get ttls
    datasep = np.squeeze(np.load(os.path.join(kilosortloc, 'datasep.npy')))/rate*1000 #get data separations 
    print(cluster.head())

    #load channel maps
    chanmap = np.load(os.path.join(kilosortloc,'channel_map.npy')) #map of the probe
    chanposition = np.load(os.path.join(kilosortloc,'channel_positions.npy'))