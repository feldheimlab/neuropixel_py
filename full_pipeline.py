#!/usr/bin/env python3
'''
Python script to run post-acquitision processing to identify single unit neurons from neuropixel data

Examples:

python full_pipeline.py -i D:/Main/Data/File # all datasets in directory will be used to use
python full_pipeline.py -i D:/Main/Data/File -d 0 2 3 # to specify which datasets to use

Saves files in a 'filtered' named folder, sister to where spikeGLX data is stored.

Kilosort files will be saved in the same folder as the filtered concatenated bin file is located.

Authors: Brian R. Mullen
Date: 2024-09-09

'''

import subprocess
import sys
import os
import shutil

import kilosort

sys.path.append('./python')

from concatenate_data import get_file_org

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
		help = 'generates the correct probemap, this can be "A", "AN", "npxl", or "linear"')
	ap.add_argument('-f', '--fps', type = int,
		default=30000,  
		help = 'frames per second for data collection')
	ap.add_argument('-d', '--datasets', type = list, 
		nargs = '+', required = False, default = None,
		help = 'list of datasets to include, if left blank all datasets in the input directory will be included')
	args = vars(ap.parse_args())

	dataloc = args['input_directory']
	saveloc = args['output_directory']
	fps = args['fps']
	probe = args['probe']
	catGTwin_loc = '..\\..\\Documents\\CatGT-win\\'
	kilosort_accessories = '..\\..\\Documents\\kilosort accessories\\'
	datasets = args['datasets']

	if probe == 'npxl':
		probe_path = '../../.kilosort/probes/4shank_NP2.0.prb'
	elif probe == 'linear':
		probe_path = '../../.kilosort/probes/NP2_kilosortChanMap.mat'
	elif probe[0] == 'A':
		probe_path = '../../.kilosort/probes/256ChanMap.mat'

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

	if not os.path.exists(saveloc):
		try:
			os.makedirs(saveloc)
			print('Making filter save location: ', saveloc)
		except Exception as e:
			print(e)

	assert os.path.exists(saveloc), 'Could not find: {}'.format(saveloc)
	print('Saving filtered data to: ', saveloc)
	
	intan = False
	CatGT = True
	folders_org, _ = get_file_org(dataloc, datasets, intan, CatGT)
	n_datasets = len(folders_org) - 1

	scripts_to_run = ['filter_concatenate',
					  'kilosort',
					  'waveform_classifier',
					  'TTL_generate']
 

	batch_script_to_run = {'filter_concatenate':{'command':[catGTwin_loc + 'runit.bat',
															'-dir='+dataloc, #location of data
															'-run=data', #what was input into spikeglx for data outputs
															'-g=0:'+ str(int(n_datasets)), #n datasets (0:1 = 2 datasets)
															'-t=0,0', #n probes
															'-prb_fld', '-ap', '-prb=0','-apfilter=butter,12,300,30000', #key parameters/filters for our data
															'-gbldmx', '-xa=0,0,1,2.5,3.5,0', '-pass1_force_ni_ob_bin',
															'-xd=2,0,-1,6,15,10', '-supercat_trim_edges',
															'-dest='+saveloc # filter save location
															], #destination directory for filter data
												 },
						   'kilosort':{'probe_loc': probe_path,
									   'settings' : {'n_chan_bin': 385,  # Number of channels in the binary file
													 'nblocks': 5,       # Enable non-rigid drift correction (use 0 for no correction)
													 'fs': fps
													 }
										},
							'waveform_classifier': {'model_loc': os.path.join(kilosort_accessories, 'waveform_model'),
													'scripts': ['./python/waveform_attributes.py', './python/classifier.py']
													} ,
							'TTL_generate': {'script': './python/concatenate_data.py',
											'input': '-filt'}

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
			
			shutil.move(os.path.join(catGTwin_loc, 'CatGT.log'), logloc)
			assert os.path.exists(binloc), 'Could not find the binary file: {}'.format(binloc) # find the filtered binary file
			print('\tFiltered data create at {}'.format(binloc))

		if script == 'kilosort':
			script_dict = batch_script_to_run[script]
			probe = kilosort.io.load_probe(script_dict['probe_loc'])
			print('Loading probe from {}'.format(script_dict['probe_loc']))

			kilosort.run_kilosort(script_dict['settings'], probe=probe, filename=binloc) 
			kilosortloc = os.path.join(saveloc, os.path.join(subdir, 'kilosort4'))
			assert os.path.exists(kilosortloc), 'Could not find the kilosort output files: {}'.format(kilosortloc) # find the filtered binary file

		if script == 'waveform_classifier':
			script_dict = batch_script_to_run[script]
			for pyscript in  script_dict['scripts']:
				result = subprocess.run([sys.executable, pyscript, '-i', kilosortloc])
				if result.returncode != 0:
					print(f"Error: {script} failed. Stopping sequence.")
					break
		if script == 'TTL_generate':
			script_dict = batch_script_to_run[script]
			result = subprocess.run([sys.executable, script_dict['script'], '-i', dataloc, script_dict['input']])
			if result.returncode != 0:
				print(f"Error: {script} failed. Stopping sequence.")
				break
		print(f"--- Finished {script} ---\n")