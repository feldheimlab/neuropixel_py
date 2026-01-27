# neuropixel_py

This is a collection of code for processing data after data acquisition. This will return the required files and identify single unit waveforms for analytic consideration.

The fullpipeline code is dependant on two external packages:

1. CatGT: filtering software, https://github.com/billkarsh/CatGT
2. Kilsort4: spike sorting software: https://github.com/MouseLand/Kilosort/tree/main

Data acquisition was performed using SpikeGLX to acquire Neuropixel recordings. The code takes into account the file organization that SpikeGLX outputs.  The input director should be the main folder that was specified during the recordings. The subdirectories will automatically be found.

The post-acquisition processing will implement the following pipeline:

1. Concatenation and filtering of raw data using CatGT, including BP filtering and DeMuxed Global average referencing (CAR)
2. Spike sorting using kilosort4 on concatenated data
3. Classify waveform outputs from Kilosort using custom RandomForest classifier
4. TTL and dataseperation identification, generating the relative times that TTLs were found in the concatenated data and the cutoff times between individual stimulation experiments.

To run the full post aquisition processing:

```python
#Concatenate all subfolders found in the data directory
python full_pipeline.py -i D:/Main/Data/File # all datasets in directory will be used to use
python full_pipeline.py -i D:/Main/Data/File -d 0 2 3 # to specify which datasets to use
```

The classifier was built based on RandomForest classifier and some of the less certain waveforms will be returned without a label. One should visualize all the data before progressing onto analysis (Phy is a built for this purpose: https://github.com/cortex-lab/phy).  The classifier will automatically use file formats that Phy will automatically detect and these classifications will be shown in Phy without need to do anything. 


