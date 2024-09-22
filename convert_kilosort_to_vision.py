#!/usr/bin/env python3
'''
Python script to take kilosort data and convert it to vision-like outputs

This script is used to convert data run through kilosort to a format that is accepted by
Feldheim lab pipeline (vision is the spikesorter that has been commonly used in the lab.)

ttlTimes.npy and datasep.npy can be in the kilosort folder or the parent directory
to the kilosort folder

Examples:

1. Concatenate all subfolders found in the data directory
python convert_kilosort_to_vision.py -i D:/Main/Data/File/kilosort4 

Saves matlab files in a sister folder called 'vision' to the kilosort folder

Authors: Brian R. Mullen
Date: 2024-09-09

'''
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.io import savemat

sys.path.append('../auditoryAnalysis/python/')
from preprocessing import probeMap, ttl_rise


def get_IDs(cluster: pd.DataFrame, 
            class_col: str, 
            matlab_version:str, 
            group: str = 'good'):
    '''
    Gets the cluster of IDs of the classification indicated
    
    Arguments:
        cluster: pd.DataFrame the indicates classification of clusters (usually saved from phy)
        class_col: column in the pd.DataFrame that the classifcation is in
        matlab_version: matlab version for saving using scipy.io
        group: which classification used, could be 'good', 'mua', 'noise'

    Returns:
        IDs: cluster ids from sorting to be included in the saved output

    '''
    IDs = cluster.loc[cluster[class_col]==group, 'cluster_id'].values
    IDs_index = cluster.loc[cluster[class_col]==group, 'cluster_id'].index

    return np.array(IDs, dtype=int),  np.array(IDs_index, dtype=int)


def ttlTimes_creation(savedir:str, 
                      ttls:np.array, 
                      matlab_version:str, 
                      rate:int):
    '''
    Gets the cluster of IDs of the classification indicated
    
    Arguments:
        savedir: save directory where the below files are saved
        ttls: array of times when the TTL was recorded
        matlab_version: matlab version for saving using scipy.io
        rate: recording rate (FPS)

    Returns:
        None

    Saves:
        ttlTimes.mat: ttltimes matlab formatted
    '''

    print('Making ttlTimes.mat file')
    savedict = {'ttlTimes':ttls}
    print('\tSaving ttlTimes.mat to: ', savedir)
    savemat(os.path.join(savedir, 'ttlTimes.mat'), savedict, format=matlab_version)


def eisummary_template(savedir:str,
                       templates:np.array, 
                       IDs:list, 
                       IDs_index:list,
                       cluster:pd.DataFrame, 
                       rate:int):
    '''
    Saves cluster waveform, with corresponding atributes
    
    Arguments:
        savedir: save directory where the below files are saved
        templates: waveforms used in the clustering
        IDs: list of identified clusters (returned by get_IDs)
        cluster: pd.DataFrame containing cluster information
        rate: recording rate (FPS)

    Returns:
        None

    Saves:
        eisummary.mat: strucutured file, formated for matlab
            contains: 
                waveform: 2-d array of the average waveform of the clustered events x neuronal IDs
                eitime: 1-d time series for one of the waveforms in ms
                maxChannels: which was the max channel
                maxAmplitudes: relative amplitude of the waveform
    '''
    
    print('Creating eisummary.mat file.')
    waveform = templates[IDs]

    center = np.nanargmin(np.min(waveform, axis=2), axis=1)
    count, bins = np.histogram(center, bins = np.arange(waveform.shape[1]))
    c = bins[np.nanargmax(count)]

    eitime = np.arange(-c, waveform.shape[1]-c)/(rate/1000)
    maxChannels  = cluster.loc[IDs_index, 'ch'].values
    maxAmplitudes = cluster.loc[IDs_index, 'Amplitude'].values

    savedict = {'waveform':waveform, 
                'eitime':eitime, 
                'maxChannels':maxChannels, 
                'maxAmplitudes':maxAmplitudes}
    print('\tSaving eisummary.mat to: ', savedir)
    savemat(os.path.join(savedir, 'eisummary.mat'), savedict, format=matlab_version)


def segmentlengths_creation(savedir: str, 
                            datasep: np.array,
                            fps:int):
    '''
    Saves the information about subsets (individual experiments) of concatenated data
    
    Arguments:
        savedir: save directory where the below files are saved
        datasep: information from when the data was concatenated acorss multiple subsets

    Returns:
        None

    Saves:
        segmentlengths.mat: strucutured file, formated for matlab
            contains: 
                data segment lengths: the length of each subset of recording
                data segment separations: the times at which they transition to a new subset
                timestamp: start time of each stimulation experiment from recording computer?
    '''
    print('Creating segmentlengths.mat file')
    timestamps = 'xxxx' # this should be the time of data collection

    try:
        print('\tFound data seperation data from python code.')
        segmentlengths = np.array(datasep.item().get('Datalength'))
        segmentseparations = np.array(datasep.item().get('Datasep'))
    except:
        print('\tFound data from matlab code.')
        segmentseparations = datasep*(1000/fps)
        segmentlengths = np.diff(datasep)*(1000/fps)

    savedict = {'timestamps':timestamps, 
                'segmentlengths':segmentlengths, 
                'segmentseparations':segmentseparations}

    print('\tSaving segmentlengths.mat to: ', savedir)
    savemat(os.path.join(savedir, 'segmentlengths.mat'), savedict, format=matlab_version)

    return savedict

def xy_creation(savedir:str, 
                IDs:list, 
                IDs_index:list,
                cluster:pd.DataFrame):
    '''
    Saves the information about subsets (individual experiments) of concatenated data
    
    Arguments:
        savedir: save directory where the below files are saved
        IDs: list of identified clusters (returned by get_IDs)
        cluster: pd.DataFrame containing cluster information

    Returns:
        xy dictionary

    Saves:
        xy.mat: strucutured file, formated for matlab
            contains: 
                x: channel x positions that recorded the highest activity
                y: channel y positions that recorded the highest activity
                xs: neuronal x positions
                ys: neuronal y positions
                ang: potential insertion angle?
    '''

    print('Making xy.mat file')
    
    chlocation = cluster.loc[IDs_index, ['x', 'y']].values
    location = cluster.loc[IDs_index, ['xs', 'ys']].values

    ang = np.zeros(IDs.shape[0])
    savedict = {'x':chlocation[:,0], 
                'y':chlocation[:,1], 
                'xs':location[:,0], 
                'ys':location[:,1], 
                'ang':ang}

    print('\tSaving xy.mat to: ', savedir)
    savemat(os.path.join(savedir, 'xy.mat'), savedict, format=matlab_version)

    return savedict


def basicinfo_creation(savedir:str, 
                       recordingregion:str='SC'):
    '''
    Saves the information about subsets (individual experiments) of concatenated data
    
    Arguments:
        savedir: save directory where the below files are saved
        recordingregion: region of electrophys recording

    Returns:
        None

    Saves:
        basicinfo.mat: metadata of recording
            contains: 
                recordingregion: location of recording
    '''
    print('Making basicinfo.mat file')
    savedict = {'recordingregion':recordingregion}
    print('\tSaving basicinfo.mat to: ', savedir)
    savemat(os.path.join(savedir, 'basicinfo.mat'), savedict, format=matlab_version)


def asdf_creation(savedir:str, 
                  IDs:list, 
                  segmentlengths:dict, 
                  spiketimes:list, 
                  spiketemplates:list, 
                  xy:dict):
    '''
    Saves the information about subsets (individual experiments) of concatenated data
    
    Arguments:
        savedir: save directory where the below files are saved
        IDs: list of identified clusters (returned by get_IDs)
        spiketimes: list of all spikes detected, given in frame of recording
        spiketemplates: corresponding list of all spikes detected, but defines the cluster assignment
        xy: dictionary from xy_creation

    Returns:
        None

    Saves:
        asdf.mat: metadata of recording for manipulation
        asdf_orig.mat: backup of the exact same mat file
            contains: 
                IDs: location of recording
                location: x, y coordinates of highest activity channels
                asdf_raw: structured array where each index corresponds to the spike times of the neurons on the IDs list
    '''
    print('Making asdf.mat file')
    asdf = np.empty((IDs.shape[0]+2,1), dtype=object)

    location = np.zeros((IDs.shape[0],2))
    # print(cluster.head())
    location[:,0]=xy['y']            
    location[:,1]=xy['x']

    totalspikes = 0 #total duration of recording
    for n, neuron in enumerate(IDs):
        try:
            asdf[n,0]=np.array(np.squeeze(spiketimes[spiketemplates==neuron]), dtype='float32')*1000/rate
            totalspikes += asdf[n,0].shape[0]
        except:
            asdf[n,0]=np.ones(1)*np.nan
    asdf[-2,0] = 1 #ms
    asdf[-1,0] = [IDs.shape[0], segmentlengths['segmentseparations'][-1]]
    savedict = {'IDs':np.array(IDs).reshape(-1,1), 
                'location':location, 
                'asdf_raw': asdf.astype(float)}
    print('\tSaving asdf.mat to: ', savedir)
    savemat(os.path.join(savedir, 'asdf.mat'), savedict, format=matlab_version)
    # savemat(os.path.join(savedir, 'asdf_orig.mat'), savedict, format=matlab_version)


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
    ap.add_argument('-p', '--probe', type = str,
        default='npxl', 
        help = 'generates the correct probemap, this can be "A", "AN", or "npxl"')
    ap.add_argument('-f', '--fps', type = int,
        default=30000,  
        help = 'frames per second for data collection')
    ap.add_argument('-ds', '--dontsave', action='store_false',
        help = 'if flagged, it will not save output graphs and initial assessements of the data')
    ap.add_argument('-g', '--group_tsv', type = str,
        default='cluster_info.tsv',  
        help = 'tsv with specified classified clusters')
    ap.add_argument('-c', '--class_col', type = str,
        default='KSLabel', 
        help = 'column name of group tsv that indicates that classification of neuronal dataset')
    args = vars(ap.parse_args())

    #necessary paths
    kilosortloc = args['input_directory']
    assert os.path.exists(kilosortloc), 'Could not find: {}'.format(kilosortloc)
    parentdir = os.path.dirname(kilosortloc)
    print(parentdir)
    savedir = os.path.join(parentdir, 'vision')
    
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    print('Vision dataset will be saved: ', savedir)

    probe = args['probe']
    rate = args['fps']
    clusterdef = args['group_tsv']
    class_col = args['class_col']
    save = args['dontsave']
    matlab_version = '5'

    if save:
        colormap = 'viridis'
        cmap_values = matplotlib.cm.get_cmap(colormap)
        print('\tGraphs, txt, and data files will be saved in the same location.')
        with open(os.path.join(savedir,'processed.txt'), 'w') as f:
            print('Processing kilosort file: ', kilosortloc, file=f)
            print('Probe used: ', probe, file=f)
            print('Rate used (FPS): ', rate, file=f)
            print('Cluster classification defined by {0} in {1} '.format(class_col, clusterdef), file=f)
    else:
        print('\tOnly data files will be saved.')

    #load kilosort data
    #spikes information
    cluster = pd.read_csv(os.path.join(kilosortloc, clusterdef), sep='\t') #class 
    #spike cluster in order of event
    spiketemplates = np.load(os.path.join(kilosortloc, 'spike_clusters.npy')) # to make asdf
    #event times
    spiketimes = np.load(os.path.join(kilosortloc, 'spike_times.npy')) # to make asdf
    #waveforms for original clusters
    templates = np.load(os.path.join(kilosortloc, 'templates.npy')) #waveforms
    spikepositions = np.squeeze(np.load(os.path.join(kilosortloc, 'spike_positions.npy'), allow_pickle=True))

    #get session/data separations times
    try:
        datasep = np.squeeze(np.load(os.path.join(kilosortloc, 'datasep.npy'), allow_pickle=True)) 
        rised = np.load(os.path.join(kilosortloc, 'ttlTimes.npy'))
        if os.path.exists(os.path.join(kilosortloc,'digital.npy')):
            print('Found the digital.npy file, continuing to determine TTLs form')
            digital = np.load(os.path.join(kilosortloc,'digital.npy'))
            rised = ttl_rise(digital, rate=rate)
    except:
        datasep = np.squeeze(np.load(os.path.join(parentdir, 'datasep.npy'), allow_pickle=True)) 
        rised = np.load(os.path.join(parentdir, 'ttlTimes.npy'))
        if os.path.exists(os.path.join(kilosortloc,'digital.npy')):
            digital = np.load(os.path.join(kilosortloc,'digital.npy'))
            rised = ttl_rise(digital, rate=rate)

    assert 'rised' in globals(), 'Could not find ttlTimes'
    assert 'datasep' in globals(), 'Could not find Datasep'

    #load channel maps
    chanposition = probeMap(probe=probe)

    #create asdf strucured array, each index is a new neuron
    neuron_clust = np.unique(spiketemplates)#determine unique clusters from the spiketemplates 
    groups = ['good', 'mua', 'noise']
    
    if save:
        with open(os.path.join(savedir,'processed.txt'), 'a') as f:
            neurons = cluster[cluster[class_col]=='good']
            neurons = pd.concat([neurons, cluster[cluster[class_col]=='mua']])

            
            print('\nProcessing kilosort file: ', kilosortloc, file=f)
            for group in groups:
                frac = len(cluster[cluster[class_col]==group])/len(cluster)
                print('{0} fraction {1}/{2}: {3}%'.format(group, len(cluster[cluster[class_col]==group]),
                                                          len(cluster), np.round(frac*100, 2)), file=f)
        
            for group in groups:
                frac = len(neurons[neurons[class_col]==group])/len(neurons)
                # if group == 'good':
                #     frac2 = len(aud_resp)/len(neurons[neurons[class_col]==group])
                #     print('good and auditory responsive fraction {0}/{1}: {2}%'.format(len(aud_resp), len(neurons[neurons[class_col]==group]), 
                #                                           np.round(frac2*100, 2)))  
                print('\t\t{0} fraction {1}/{2}: {3}%'.format(group, len(neurons[neurons[class_col]==group]), 
                                                         len(neurons), np.round(frac*100, 2)))
    #visualize where the clusters are located across the probe
    if probe == 'npxl':
        ylim = [-50,775]
    else:
        ylim = [-50,1150]

    cluster['group_c'] = cluster[class_col]
    for i, group in enumerate(groups):
        cluster.loc[cluster[cluster[class_col]==group].index, 'group_c'] = int(i)

    for c_id in cluster['cluster_id']:        
        points = spikepositions[spiketemplates==c_id]
        cluster.loc[cluster['cluster_id']==c_id, 'xs']=np.mean(points[:,0])
        cluster.loc[cluster['cluster_id']==c_id, 'ys']=np.mean(points[:,1])
        pos = np.squeeze(chanposition[cluster.loc[cluster['cluster_id'] == c_id, 'ch'].values,:])
        cluster.loc[cluster['cluster_id']==c_id, 'x']=pos[1]
        cluster.loc[cluster['cluster_id']==c_id, 'y']=pos[0]

    if save:
        fig, axs = plt.subplots(1,4, figsize = (10,5), sharey=True)

        for j, group in enumerate(groups):
            chanposition = probeMap(probe=probe)
            axs[j].scatter(chanposition[:,1], chanposition[:,0], facecolors='None', edgecolors='k')
            clust = cluster[cluster[class_col]==group].copy()
            for index in clust.index:        
                pos = chanposition[clust.loc[index, 'ch']]
                pos[1] = clust.loc[index, 'xs']
                pos[0] = clust.loc[index, 'ys']
                color= clust.loc[index, 'group_c']
                axs[j].scatter(pos[1], pos[0], color=cmap_values(color/(len(groups)-1)), alpha=0.25, vmin=0, vmax=2)
                axs[j].set_ylim(ylim)
                
        axs[0].set_ylabel('depth (um)')
        axs[1].set_xlabel('span (um)')
        axs[3].set_xlabel('FR')

        im1 = axs[3].scatter(cluster['fr'], cluster['depth'],  c=cluster['group_c'], cmap=colormap, alpha=0.25, vmin=0, vmax=2)
        axs[3].set_ylim(ylim)
        divider = make_axes_locatable(axs[3])

        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im1, cax =cax , orientation='vertical')
        cbar.set_ticks([0,1,2])
        cbar.set_ticklabels(groups)
        plt.savefig(os.path.join(savedir, 'cluster_loc_group_based.png'), dpi=300)
        # plt.show()

    IDs, IDs_index = get_IDs(cluster, class_col, matlab_version=matlab_version, group='good')
    ttlTimes_creation(savedir, ttls=rised, matlab_version=matlab_version, rate=rate)
    
    xy = xy_creation(savedir, IDs, IDs_index, cluster)

    eisummary_template(savedir, templates, IDs, IDs_index, cluster, rate=rate)

    segmentlengths = segmentlengths_creation(savedir, datasep, fps=rate)
    
    basicinfo_creation(savedir, recordingregion='SC')

    asdf_creation(savedir, IDs, segmentlengths, spiketimes, spiketemplates, xy)
