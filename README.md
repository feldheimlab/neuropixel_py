# neuropixel_py

Post-acquisition processing pipeline for Neuropixel and Intan electrophysiology recordings. Filters and concatenates raw data, runs Kilosort4 spike sorting, classifies waveforms using a trained ML model, and converts results to VISION-compatible formats for downstream analysis.

## Dependencies

The full pipeline depends on two external packages:

1. **CatGT**: Filtering software — https://github.com/billkarsh/CatGT
2. **Kilosort4**: Spike sorting software — https://github.com/MouseLand/Kilosort

## Installation

```bash
pip install -r requirements.txt
```

Cross-repo dependency: the [auditoryAnalysis](https://github.com/your-org/auditoryAnalysis) repository must be cloned as a sibling directory (`../auditoryAnalysis/`) for TTL extraction and probe map functions.

## Usage

### Full pipeline

Runs CatGT filtering, Kilosort4 sorting, waveform classification, and TTL extraction in sequence:

```bash
python full_pipeline.py -i /path/to/raw/data -p npxl -f 30000
```

| Argument | Description | Default |
|----------|-------------|---------|
| `-i` | Path to raw data directory (required) | — |
| `-o` | Output directory for filtered data | `{input}/filtered` |
| `-p` | Probe type: `npxl`, `linear`, or `A`-prefixed | `npxl` |
| `-f` | Sampling rate in Hz | `30000` |
| `-d` | Dataset indices to include | all |

### Standalone scripts

Each script in `python/` can also be run independently:

```bash
# Concatenate raw binary data
python python/concatenate_data.py -i /path/to/data -con

# Extract TTL times from CatGT-filtered data
python python/concatenate_data.py -i /path/to/data -filt

# Recompute waveform templates (slow — reads raw data)
python python/concatenate_data.py -i /path/to/filtered/subdir -w

# FFT spectral analysis of raw data
python python/concatenate_data.py -i /path/to/data -fft

# Extract waveform attributes for classification
python python/waveform_attributes.py -i /path/to/kilosort4

# Classify waveforms using trained model
python python/classifier.py -i /path/to/kilosort4

# Convert kilosort output to VISION .mat format
python python/convert_kilosort_to_vision.py -i /path/to/kilosort4 -p npxl
```

## Pipeline Overview

```
Raw SpikeGLX recordings (.bin)
  → CatGT filtering & concatenation
  → Kilosort4 spike sorting
  → Waveform attribute extraction
  → ML waveform classification (good / mua / noise)
  → TTL extraction from filtered data
  → VISION .mat format conversion
```

## Project Structure

```
neuropixel_py/
├── full_pipeline.py              # Main pipeline orchestrator
├── config.py                     # Pipeline configuration class
├── python/
│   ├── __init__.py
│   ├── concatenate_data.py       # Data concatenation, TTL extraction, waveform summary, FFT
│   ├── waveform_attributes.py    # Waveform feature extraction from templates
│   ├── classifier.py             # ML waveform classification
│   └── convert_kilosort_to_vision.py  # Kilosort → VISION .mat conversion
├── requirements.txt
└── README.md
```

## Output

The pipeline produces filtered data and kilosort output in the save directory, plus VISION-compatible `.mat` files in a `vision/` subdirectory:

- `datasep.npy` / `ttlTimes.npy` — Segment boundaries and TTL event times
- `kilosort4/` — Spike sorting results, cluster classifications, waveform attributes
- `vision/` — MATLAB-formatted files (`asdf.mat`, `ttlTimes.mat`, `eisummary.mat`, `segmentlengths.mat`, `xy.mat`, `basicinfo.mat`)
