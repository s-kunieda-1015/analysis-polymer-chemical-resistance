# Data Directory

This directory contains datasets required for the chemical resistance prediction analysis.

## Data Files

### Tracked by Git

- `chemical_resistance_metadata.json` - Metadata including polymer/solvent naming dictionaries

### Not tracked by Git (download from Zenodo)

The following files should be downloaded and placed in this directory:

- `chemical_resistance_dataset.csv` - Main chemical resistance experimental data
- `pe_solvent_dataset.csv` - Polyethylene virtual solvent dataset
- `polymer_mpk_dataset.csv` - Virtual polymer dataset with MPK solvent

### Dataset View Subdirectory

The `dataset_view/` subdirectory contains formatted versions for easier viewing.

## Data Sources

### polymer_mpk_dataset.csv
This dataset is based on the [PolyOmics](https://huggingface.co/datasets/yhayashi1986/PolyOmics) dataset.
See the related paper: [arXiv:2511.11626](https://arxiv.org/abs/2511.11626)

### pe_solvent_dataset.csv
The solvent SMILES data is derived from [HSPiP (Hansen Solubility Parameters in Practice)](https://www.pirika.com/wp/chemistry-at-pirika-com/hsp/how2buy).
Note: The HSP values themselves are not used in this study.

## Data Availability

These datasets are available from Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17775728.svg)](https://doi.org/10.5281/zenodo.17775728)

**DOI**: [10.5281/zenodo.17775728](https://doi.org/10.5281/zenodo.17775728)

> **Note**: The dataset is currently restricted. Please request access through Zenodo.

## Note

Large data files are excluded from version control due to their size.
Please download the datasets from Zenodo before running the analysis.

