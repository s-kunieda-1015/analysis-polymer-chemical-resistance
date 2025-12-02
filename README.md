# Machine Learning for Polymer Chemical Resistance to Organic Solvents

This repository contains the code and analysis for predicting chemical resistance of polymer-solvent combinations using machine learning models.

## Paper Citation

> Shogo Kunieda, Mitsuru Yambe, Hiromori Murashima, Takeru Nakamura, Toshiaki Shintani, Hitoshi Kamijima, Yoshihiro Hayashi, Yosuke Hanawa, Ryo Yoshida. "Machine Learning for Polymer Chemical Resistance to Organic Solvents." arXiv:2509.05344 (2025).

**arXiv**: [https://arxiv.org/abs/2509.05344](https://arxiv.org/abs/2509.05344)

### Abstract

Predicting the chemical resistance of polymers to organic solvents is a longstanding challenge in materials science, with significant implications for sustainable materials design and industrial applications. Here, we address the need for interpretable and generalizable frameworks to understand and predict polymer chemical resistance beyond conventional solubility models. We systematically analyze a large dataset of polymer solvent combinations using a data-driven approach. Our study reveals that polymer crystallinity and density, as well as solvent polarity, are key factors governing chemical resistance, and that these trends are consistent with established theoretical models. These findings provide a foundation for rational screening and design of polymer materials with tailored chemical resistance, advancing both fundamental understanding and practical applications.

## Repository Structure

```
analysis-polymer-chemical-resistance/
├── src/
│   ├── main_analysis.py          # Main analysis script
│   ├── main_analysis.ipynb       # Jupyter notebook (with execution results)
│   └── convert_to_notebook.py    # Script to generate notebook from .py
├── utils/                        # Utility modules
│   ├── config.py                 # Configuration and paths
│   ├── data_preprocessing.py     # Data loading functions
│   ├── plotting.py               # Visualization functions
│   ├── validation.py             # Cross-validation functions
│   ├── train.py                  # Model training functions
│   ├── metrics.py                # Evaluation metrics
│   └── mic.py                    # MIC calculation
├── data/                         # Data files (download from Zenodo)
├── output/                       # Generated outputs
│   ├── LOGOCV/                   # Cross-validation results
│   ├── Main/                     # Main figures
│   ├── Sub/                      # Supplementary figures
│   └── Model/                    # Trained models
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Data Download

The datasets required for this analysis are available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17775728.svg)](https://doi.org/10.5281/zenodo.17775728)

**Zenodo DOI**: [10.5281/zenodo.17775728](https://doi.org/10.5281/zenodo.17775728)

Download the following files and place them in the `data/` directory:

| File | Description | Size |
|------|-------------|------|
| `chemical_resistance_dataset.csv` | Main experimental dataset (2231 records) | 21.02 MB |
| `pe_solvent_dataset.csv` | PE solvent dataset (9828 records) | 68.12 MB |
| `polymer_mpk_dataset.csv` | Virtual polymer dataset (40971 records) | 251.52 MB |
| `general_polymers_with_sp_abbe_dynamic-dielectric.csv` | General polymer properties (Source from [PolyOmics](https://huggingface.co/datasets/yhayashi1986/PolyOmics) data) | 190 MB |

> **Note**: The solvent SMILES data in `pe_solvent_dataset.csv` is derived from [HSPiP (Hansen Solubility Parameters in Practice)](https://www.pirika.com/wp/chemistry-at-pirika-com/hsp/how2buy). The HSP values themselves are not used in this study.

> **Note**: The `polymer_mpk_dataset.csv` is derived from the [PolyOmics](https://huggingface.co/datasets/yhayashi1986/PolyOmics) dataset (see related paper: [arXiv:2511.11626](https://arxiv.org/abs/2511.11626)). This file contains virtual polymers with no missing values for the physical properties used in chemical resistance prediction, with descriptors pre-calculated. Note that this file does not include the property values from PolyOmics; those are restored at runtime by joining with `general_polymers_with_sp_abbe_dynamic-dielectric.csv`.

> **Note**: The `general_polymers_with_sp_abbe_dynamic-dielectric.csv` file is available from the [PolyOmics dataset on Hugging Face](https://huggingface.co/datasets/yhayashi1986/PolyOmics). Download and place it in the `data/` directory.

> **Note**: `polymer_mpk_dataset.csv` and `general_polymers_with_sp_abbe_dynamic-dielectric.csv` are joined at runtime to restore the RadonPy columns (`_radonpy_polymer`). The join is performed using the `smiles_polymer` column (with `[*]` converted to `*`) matched against the `smiles_list` column. Rows that cannot be matched are dropped since they lack the required RadonPy data for prediction.

## Installation

### Requirements

- Python 3.9+
- Required packages listed in `requirements.txt`

### Setup

1. Clone this repository:
```bash
git clone https://github.com/s-kunieda-1015/analysis-polymer-chemical-resistance.git
cd analysis-polymer-chemical-resistance
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download data from Zenodo and place in `data/` directory.

## Usage

### Generate Jupyter Notebook

To convert the Python script to a Jupyter notebook for interactive viewing:

```bash
cd src
python convert_to_notebook.py
```

This creates `main_analysis.ipynb` which can be opened in Jupyter.

### Run Analysis

**Option 1: Run as Python script**
```bash
cd src
python main_analysis.py
```

**Option 2: Run as Jupyter notebook**
```bash
cd src
jupyter notebook main_analysis.ipynb
```

### Command Line Options

```bash
python main_analysis.py [OPTIONS]

Options:
  --skip-resistance-training    Skip resistance model training (use saved model)
  --skip-chi-training           Skip chi parameter model training (use saved model)
  --skip-crystallinity-training Skip crystallinity model training (use saved results)
  --skip-logocv                 Skip LOGOCV validation (use saved results)
  --skip-mic                    Skip MIC calculation (use saved results)
  --n-jobs N                    Number of parallel jobs (default: 8)
```

**Quick execution (use pre-computed results):**
```bash
python main_analysis.py --skip-resistance-training --skip-chi-training --skip-crystallinity-training --skip-logocv --skip-mic
```

## Generated Figures

### Main Figures
- **Figure 1**: Chemical Resistance Heatmap
- **Figure 3**: LOGOCV Cross-Validation Results
- **Figure 4**: Feature Importance Analysis (MIC)
- **Figure 5**: Resistance Prediction Results
- **Figure 6**: Crystallinity Effect on Resistance
- **Figure 7a-d**: Chi Parameter Analysis
- **Figure 8**: Comprehensive Model Comparison
- **Figure 9**: Crystallinity Parameter Analysis

### Supplementary Figures
- **S1**: Resistance Distribution by Polymer
- **S2**: Cluster Heatmaps
- **S3**: ROC Curves and Confusion Matrices
- **S4**: Cross-Validation Performance
- **S5**: Force Field Parameter Analysis
- **S6**: Chi Parameter Histogram by Polymer

## Models

Three machine learning models are trained:

| Model | Algorithm | Purpose |
|-------|-----------|---------|
| Chemical Resistance | GBDT | Predict polymer-solvent resistance |
| Crystallinity | Random Forest | Classify polymer crystallinity |
| Chi Parameter | XGBoost | Predict chi interaction parameter |

Pre-trained models are saved in `output/Model/` after the first execution.

## Acknowledgments

- The notebook conversion script (`convert_to_notebook.py`) was developed with reference to [nb2py](https://github.com/tactical-k/nb2py).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Shogo Kunieda - kuniedascreen@gmail.com

