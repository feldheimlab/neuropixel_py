# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neuropixel post-acquisition processing pipeline (Feldheim Lab, UC Santa Cruz). Takes raw Neuropixel or Intan electrophysiology recordings, filters/concatenates them, runs Kilosort spike sorting, classifies waveforms via a trained ML model, and converts output to VISION-compatible formats for downstream auditory analysis.

## Running the Pipeline

### Full pipeline (requires CatGT + Kilosort)

```bash
python full_pipeline.py -i /path/to/raw/data -p npxl -f 30000
```

Key arguments:
- `-i`: Path to raw data directory containing SpikeGLX recording subfolders
- `-o`: Output directory for filtered data (default: `{input}/filtered`)
- `-p`: Probe type (`npxl`, `linear`, or `A`-prefixed for Intan)
- `-f`: Sampling rate in Hz (default 30000)
- `-d`: Specific dataset indices to include (default: all)

### Standalone scripts

```bash
# Concatenate raw data
python python/concatenate_data.py -i /path/to/data -con

# Get TTL times from CatGT-filtered data
python python/concatenate_data.py -i /path/to/data -filt

# Recompute waveform templates from raw data
python python/concatenate_data.py -i /path/to/filtered/subdir -w

# Extract waveform attributes for classification
python python/waveform_attributes.py -i /path/to/kilosort4

# Classify waveforms using trained ML model
python python/classifier.py -i /path/to/kilosort4

# Convert kilosort output to VISION .mat format
python python/convert_kilosort_to_vision.py -i /path/to/kilosort4 -p npxl
```

## Architecture

### Pipeline Flow

```
Raw SpikeGLX recordings (.bin files)
  → CatGT filtering & concatenation (external, via subprocess)
  → Kilosort4 spike sorting (via kilosort Python API)
  → Waveform attribute extraction [waveform_attributes.py]
  → ML waveform classification [classifier.py]
  → TTL extraction from CatGT output [concatenate_data.py]
  → VISION format conversion [convert_kilosort_to_vision.py]
```

### Module Responsibilities

- **`full_pipeline.py`** — Main entry point. Argument parsing, orchestrates CatGT filtering, Kilosort sorting, waveform classification, and TTL extraction as sequential subprocess/API calls.
- **`config.py`** — `configs` class centralizing all pipeline parameters: CatGT command construction, Kilosort settings, probe paths, script locations.
- **`python/concatenate_data.py`** — Data concatenation (`concatentate_npx_data`, `concatentate_intan_data`), TTL extraction from raw (`ttl_npx_data`) or CatGT-filtered (`ttl_npx_filtered_data`, `datasep_ttltimes_file_update`) data, waveform summary computation (`make_waveform_summary`), and FFT analysis (`fft_raw_data`).
- **`python/waveform_attributes.py`** — `waveform_features` class: normalizes templates, identifies active channels, extracts peak counts/locations/amplitudes, and computes waveform duration metrics (AP rise, fall, AHP return). Saves `cluster_attribute_data.tsv`.
- **`python/classifier.py`** — Loads a pre-trained RandomForest model to classify kilosort clusters as 'good', 'mua', or 'noise' based on waveform attributes. Updates `cluster_group.tsv`.
- **`python/convert_kilosort_to_vision.py`** — Converts kilosort output to VISION-compatible `.mat` files: `asdf.mat`, `ttlTimes.mat`, `eisummary.mat`, `segmentlengths.mat`, `xy.mat`, `basicinfo.mat`.

### Import Pattern

`python/` is a package (has `__init__.py`). The main pipeline uses `from python.concatenate_data import ...`. Standalone scripts in `python/` use `sys.path.append` for cross-repo dependencies (`auditoryAnalysis/python/preprocessing`). Run the pipeline from the repository root.

### Cross-Repo Dependencies

`concatenate_data.py` and `convert_kilosort_to_vision.py` import from the sibling `auditoryAnalysis` repository (`preprocessing.ttl_rise`, `preprocessing.probeMap`). These paths are set via `sys.path.append('../auditoryAnalysis/python/')`.

## Key Data Structures

- **Binary files (.bin)**: Raw int16 recordings, 385 channels (384 neural + 1 digital), memory-mapped for processing.
- **Kilosort outputs**: `spike_times.npy`, `spike_clusters.npy`, `templates.npy`, `cluster_info.tsv`, etc.
- **VISION .mat files**: MATLAB-compatible structured arrays for the auditory analysis pipeline.
- **datasep**: Dict with `Datasep` (segment boundary times in ms), `Datalength` (segment durations), `Timestamp`.

## Output Structure

Results from the full pipeline appear in the filtered data directory alongside kilosort output:
```
filtered/
└── {subdir}/
    ├── *.bin (filtered concatenated data)
    ├── datasep.npy, ttlTimes.npy
    └── kilosort4/
        ├── spike_times.npy, spike_clusters.npy, templates.npy, ...
        ├── cluster_info.tsv, cluster_group.tsv, cluster_attribute_data.tsv
        └── stored_norm_waveform.npy
vision/
    ├── asdf.mat, ttlTimes.mat, eisummary.mat
    ├── segmentlengths.mat, xy.mat, basicinfo.mat
    └── cluster_loc_group_based.png
```

## Dependencies

Install with `pip install -r requirements.txt`. External: CatGT (Windows batch filtering tool). Cross-repo: `auditoryAnalysis/python/preprocessing.py`.

## Development Notes

- Tabs are used for indentation throughout the codebase.
- No automated tests. Verify with `python3 -m py_compile <file>`.
- The project uses bare `except` clauses in several places for error handling.
