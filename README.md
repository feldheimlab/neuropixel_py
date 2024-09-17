# neuropixel_py


To get help from the terminal
```python
#Concatenate all subfolders found in the data directory
python concatenate_data.py -h
```

To concatentate, get TTL event times, look at the spectral information from the neuropixel data add the appropriate flag to the command line (multiple flags can be used at once):

```python
#Concatenate all subfolders found in the data directory
python concatenate_data.py -i D:/Main/Data/File -c

#Concatenate only the datasets indicated that are found in the data directory
python concatenate_data.py -i D:/Main/Data/File -d 0 2 3 -c

#Get TTLS from all subfolders found in the data directory
python concatenate_data.py -i D:/Main/Data/File -t

#Get spectral infromation from all subfolders found in the data directory
python concatenate_data.py -i D:/Main/Data/File -fft

```

Once kilosort has been run and a subfolder has been created, you can re-make the tempates.  After using phy to manager clusters identity and merging/splitting the templates that are saved may not be the current waveform summaries.  To update the templates used in the machine learning clustering of each neuron, you can run the following code (this does take a long time):  

```python

#Concatenate only the datasets indicated that are found in the data directory
python concatenate_data.py -i D:/Main/Data/File -w

```
