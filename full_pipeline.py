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

import subprocess
import sys
import os

import kilosort


# List of scripts to run in order
if __name__ == '__main__':

	import argparse
	import time
	import datetime

	# Argument Parsing
	# -----------------------------------------------
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--input_directory', type = str,
		required = True, 
		help = 'path to data collected from neuropixels')
	ap.add_argument('-o', '--output_directory', type = str,
		required = False, default=None,
		help = 'path to output filtered data and kilsort location')
	ap.add_argument('-p', '--probe', type = str,
		default='npxl', 
		help = 'generates the correct probemap, this can be "A", "AN", or "npxl"')
	ap.add_argument('-f', '--fps', type = int,
		default=30000,  
		help = 'frames per second for data collection')
	args = vars(ap.parse_args())

	dataloc = args['input_directory']
	saveloc = args['output_directory']
	fps = args['fps']
	catGTwin_loc = '..\\..\\Documents\\CatGT-win\\'
	kilosort_accessories = '..\\..\\Documents\\kilosort accessories\\'

	print('Starting pipeline:')
	print('Assume the location of the following:')
	assert os.path.exists(catGTwin_loc), 'Could not find: {}'.format(catGTwin_loc)
	print('\tLocation of CatGT package {}'.format(catGTwin_loc))
	assert os.path.exists(kilosort_accessories), 'Could not find: {}'.format(kilosort_accessories)
	print('\tLocation of kilosort accessories {}'.format(kilosort_accessories))

	assert os.path.exists(dataloc), 'Could not find: {}'.format(dataloc)
	print('Found data location: ', dataloc)
	
	if saveloc == None:
		# dirname = os.path.dirname(dataloc)
		saveloc = os.path.join(dataloc, 'filtered')

	if not os.path.exists(saveloc):
		try:
			os.makedirs(saveloc)
			print('Making filter save location: ', saveloc)
		except Exception as e:
			print(e)

	assert os.path.exists(saveloc), 'Could not find: {}'.format(saveloc)
	print('Saving filtered data to: ', saveloc)
	
	scripts_to_run = [
		'filter_concatenate',
		'kilosort',
		'assignemnt'
	]


	batch_script_to_run = {'filter_concatenate':{'command':[catGTwin_loc + 'runit.bat',
															'-dir='+dataloc, #location of data
															'-run=data', #what was input into spikeglx for data outputs
															'-g=0:1', #n datasets (0:1 = 2 datasets)
															'-t=0,0', #n probes
												 			'-prb_fld', '-ap', '-prb=0','-apfilter=butter,12,300,30000', #key parameters/filters for our data
												 			'-gbldmx', '-xa=0,0,1,2.5,3.5,0', '-pass1_force_ni_ob_bin',
												 			'-dest='+saveloc # filter save location
												 			], #destination directory for filter data
												 },
						   'kilosort':{'probe_loc': '../../.kilosort/probes/NP2_kilosortChanMap.mat',
						   			   'settings' : {'n_chan_bin': 385,  # Number of channels in the binary file
    											     'nblocks': 5,       # Enable non-rigid drift correction (use 0 for no correction)
   													 'fs': fps
   													 }
   										},
							'waveform_classifier': {'modelloc': 'modelloc'} 

							}

	# Run each script sequentially
	for script in scripts_to_run:
		command_dict = batch_script_to_run[script]

		print(f"--- Running {script} ---")
		if script == 'filter_concatenate':
			assert os.path.exists(command_dict['command'][0]), '\tCould not find: {}'.format(command_dict['command'])
			p = subprocess.Popen(command_dict['command'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
			output, errors = p.communicate()
			p.wait()

			walkresults = list(os.walk(saveloc))
			subdir = walkresults[0][1][0]
			subdir_files = walkresults[1][2]
			for file in subdir_files:
				if file.endswith('.bin'):
					binfile = file
			binloc = os.path.join(saveloc, os.path.join(subdir, binfile))
			
			logloc = os.path.join(saveloc, 'CatGT.log') # move log file over to filtered location
			if  os.path.exists(logloc):
				print('\tDeleting log from previous run {}'.format(logloc))
				os.remove(logloc)
			os.rename(os.path.join(catGTwin_loc, 'CatGT.log'), logloc)
			
			assert os.path.exists(binloc), 'Could not find the binary file: {}'.format(binloc) # find the filtered binary file
			print('\tFiltered data create at {}'.format(binloc))

		if script == 'kilosort':
			settings_dict = batch_script_to_run[script]
			
			probe = kilosort.io.load_probe(settings_dict['probe_loc'])
			print('Loading probe from {}'.format(settings_dict['probe_loc']))

			kilosort.run_kilosort(settings_dict['settings'], probe=probe, filename=binloc)

		if script == 'waveform_classifier':
			print('data here')
		#     # subprocess.run waits for the script to complete
		#     result = subprocess.run([sys.executable, script])
		
		# Optional: check if the previous script was successful
		# if errors.returncode != 0:
		# 	print(f"Error: {script} failed. Stopping sequence.")
		# 	break
		print(f"--- Finished {script} ---\n")