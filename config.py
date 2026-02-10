#!/usr/bin/env python3
'''
Config file required to run "full_pipeline.py"

Authors: Brian R. Mullen
Date: 2024-09-09

'''

import os


class configs():
	def __init__(self, dataloc, saveloc, probe, n_datasets, fps):
		self.catGTwin_loc = '..\\..\\Documents\\CatGT-win\\'
		self.kilosort_accessories = '..\\..\\Documents\\kilosort accessories\\'

		# print('Assume the location of the following:')
		# assert os.path.exists(self.catGTwin_loc), 'Could not find: {}'.format(self.catGTwin_loc)
		# print('\tLocation of CatGT package {}'.format(self.catGTwin_loc))
		# assert os.path.exists(self.kilosort_accessories), 'Could not find: {}'.format(self.kilosort_accessories)
		# print('\tLocation of kilosort accessories {}'.format(self.kilosort_accessories))

		self.dataloc = dataloc
		self.saveloc = saveloc
		self.binloc = None
		self.probe = probe
		self.fps = fps
		self.n_datasets = n_datasets

		if self.probe == 'npxl':
			self.probe_path = '../../.kilosort/probes/4shank_NP2.0.prb'
			self.npxl = True
			self.intan = False
		elif self.probe == 'linear':
			self.probe_path = '../../.kilosort/probes/NP2_kilosortChanMap.mat'
			self.npxl = True
			self.intan = False
		elif self.probe[0] == 'A':
			self.probe_path = '../../.kilosort/probes/256ChanMap.mat'
			self.npxl = False
			self.intan = True
		else:
			self.probe_path = None
			print('Unknown probe type. The script cannot run kilosort unless we know the probe.')

		self.batch_script_to_run = {'filter_concatenate':{'command':[self.catGTwin_loc + 'runit.bat',
															'-dir='+self.dataloc, #location of data
															'-run=data', #what was input into spikeglx for data outputs
															'-g=0:'+ str(int(self.n_datasets)), #n datasets (0:1 = 2 datasets)
															'-t=0,0', #n probes
															'-prb_fld', '-ap', '-prb=0','-apfilter=butter,12,300,30000', #key parameters/filters for our data
															'-gbldmx', '-xa=0,0,1,2.5,3.5,0', '-pass1_force_ni_ob_bin',
															'-xd=2,0,-1,6,15,10', '-supercat_trim_edges',
															'-dest='+saveloc # filter save location
															], #destination directory for filter data
												 },
								   'kilosort':{'probe_loc': self.probe_path,
											   'settings' : {'n_chan_bin': 385,  # Number of channels in the binary file
															 'nblocks': 5,       # Enable non-rigid drift correction (use 0 for no correction)
															 'fs': self.fps
															 }
												},
									'waveform_classifier': {'model_loc': os.path.join(self.kilosort_accessories, 'waveform_model'),
															'scripts': ['./python/waveform_attributes.py', './python/classifier.py']
															} ,
									'TTL_generate': {'script': './python/concatenate_data.py',
													'input': '-filt'}

									}
		self.scripts_to_run = list(self.batch_script_to_run.keys())									
		# self.scripts_to_run = ['filter_concatenate',
		# 					  'kilosort',
		# 					  'waveform_classifier',
		# 					  'TTL_generate']
		
	def write_attributes(self):
		with open(os.path.join(self.saveloc, 'postprocessing_steps.txt'), 'w') as f:
		    for keys, values in vars(self).items():
		    	if isinstance(values, dict):
		    		f.write(f"{keys}\n")
		    		for key, value in values.items():
		    			f.write(f"{key}:{value}\n\n")
		    	else:
		        	f.write(f"{keys}:{values}\n\n")
