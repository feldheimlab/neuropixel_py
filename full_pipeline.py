#!/usr/bin/env python3
'''
Python script to run post-acquitision processing to identify single unit neurons from neuropixel data

Examples:

python full_pipeline.py -i D:/Main/Data/File # all datasets in directory will be used to use
python full_pipeline.py -i D:/Main/Data/File -d 0 2 3 # to specify which datasets to use

Saves files in a 'filtered' named folder, sister to where spikeGLX data is stored.

Kilosort files will be saved in the same folder as the filtered concatenated bin file is located.

Authors: Brian R. Mullen
Date: 2026-02-04

'''

import subprocess
import sys
import os
import shutil

import kilosort

from python.concatenate_data import get_file_org
from config import configs

def ask_yes_or_no(msg: str) -> bool:
    while True:
        if (user := input(msg).lower()) in ('y', 'yes'):
            return True
        if user in ('n' ,'no'):
            return False
        print('Invalid input. Please try again...')


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
	datasets = args['datasets']

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

	folders_org, _ = get_file_org(dataloc, datasets, intan=False, CatGT=True)
	n_datasets = len(folders_org) - 1
	
	config = configs(dataloc, saveloc, probe, n_datasets, fps)

	walkresults = list(os.walk(config.saveloc))
	try:
		subdir = walkresults[0][1][0]
		subdir_files = walkresults[1][2]
		for file in subdir_files:
			if file.endswith('.bin'):
				binfile = file
		config.binloc = os.path.join(config.saveloc, os.path.join(subdir, binfile))
		kilosortloc = os.path.join(saveloc, os.path.join(subdir, 'kilosort4'))
	except:
		config.binloc = None
		kilosortloc = None

	if os.path.exists(config.binloc):
		print('Found concatenated bin file:', config.binloc)
		if not ask_yes_or_no('Would you like to redo the filter and concatenation step of the data processing? [Y/N]: '):
			config.scripts_to_run.remove('filter_concatenate')
	else:
		print('No concatenated binfile has been found.')
	
	if os.path.exists(kilosortloc):
		print('Found kilsort file:', kilosortloc)
		if not ask_yes_or_no('Would you like to redo spikesorting with kilsort? [Y/N]: '):
			config.scripts_to_run.remove('kilosort')
	else:
		print('No kilosort folder has been found.')	

	print('Starting pipeline:')

	# Run each script sequentially
	for script in config.scripts_to_run:
		print(f"--- Running {script} ---")
		if script == 'filter_concatenate':
			command_dict = config.batch_script_to_run[script]
			assert os.path.exists(command_dict['command'][0]), '\tCould not find: {}'.format(command_dict['command'][0])
			p = subprocess.Popen(command_dict['command'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
			output, errors = p.communicate()
			p.wait()
			walkresults = list(os.walk(config.saveloc))
			subdir = walkresults[0][1][0]
			subdir_files = walkresults[1][2]
			for file in subdir_files:
				if file.endswith('.bin'):
					binfile = file
			config.binloc = os.path.join(config.saveloc, os.path.join(subdir, binfile))
			
			logloc = os.path.join(config.saveloc, 'CatGT.log') # move log file over to filtered location
			if  os.path.exists(logloc):
				print('\tDeleting log from previous run {}'.format(logloc))
				os.remove(logloc)
			
			if os.path.exists(os.path.join(config.catGTwin_loc, 'CatGT.log')):
				shutil.move(os.path.join(config.catGTwin_loc, 'CatGT.log'), logloc)
			
			assert os.path.exists(config.binloc), 'Could not find the binary file: {}'.format(config.binloc) # find the filtered binary file
			print('\tFiltered data create at {}'.format(config.binloc))
			
		if 'filter_concatenate' not in config.scripts_to_run:	
			walkresults = list(os.walk(config.saveloc))
			subdir = walkresults[0][1][0]
			subdir_files = walkresults[1][2]
			for file in subdir_files:
				if file.endswith('.bin'):
					binfile = file
			config.binloc = os.path.join(config.saveloc, os.path.join(subdir, binfile))

			assert os.path.exists(config.binloc), 'Could not find the binary file: {}'.format(config.binloc) # find the filtered binary file
			print('\tFiltered data located at {}'.format(config.binloc))

		if script == 'kilosort':
			kilosortloc = os.path.join(saveloc, os.path.join(subdir, 'kilosort4'))

			script_dict = config.batch_script_to_run[script]
			probe = kilosort.io.load_probe(script_dict['probe_loc'])
			print('Loading probe from {}'.format(script_dict['probe_loc']))

			kilosort.run_kilosort(script_dict['settings'], probe=probe, filename=config.binloc) 

			kilosortloc = os.path.join(saveloc, os.path.join(subdir, 'kilosort4'))
			assert os.path.exists(kilosortloc), 'Could not find the kilosort output files: {}'.format(kilosortloc) # find the filtered binary file

		if 'kilosort' not in config.scripts_to_run:
			kilosortloc = os.path.join(saveloc, os.path.join(subdir, 'kilosort4'))
			assert os.path.exists(kilosortloc), 'Could not find the kilosort output files: {}'.format(kilosortloc) # find the filtered binary file
		
		if script == 'waveform_classifier':
			script_dict = config.batch_script_to_run[script]
			for pyscript in  script_dict['scripts']:
				print(pyscript)
				result = subprocess.run([sys.executable, pyscript, '-i', kilosortloc])
				if result.returncode != 0:
					print(f"Error: {script} failed. Stopping sequence.")
					break
		
		if script == 'TTL_generate':
			script_dict = config.batch_script_to_run[script]
			print(script_dict['script'])

			result = subprocess.run([sys.executable, script_dict['script'], '-i', config.dataloc, script_dict['input']])
			if result.returncode != 0:
				print(f"Error: {script} failed. Stopping sequence.")
				break
		
		print(f"--- Finished {script} ---\n")
		config.write_attributes()
