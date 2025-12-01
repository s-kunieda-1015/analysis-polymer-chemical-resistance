# Chemical Resistance Prediction Analysis

This repository contains the code and analysis for predicting chemical resistance of polymer-solvent combinations using machine learning models.

## Paper Citation

> [Paper citation will be added upon publication]

## Repository Structure

```
analysis-polymer-chemical-resistance/
├── src/
│   ├── main_analysis.py          # Main analysis script
│   └── convert_to_notebook.py    # Script to convert .py to Jupyter notebook
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

**DOI**: [DOI will be added upon publication]

Download the following files and place them in the `data/` directory:

| File | Description | Size |
|------|-------------|------|
| `chemical_resistance_dataset.csv` | Main experimental dataset (2231 records) | ~20 MB |
| `chemical_resistance_metadata.json` | Metadata including name mappings | ~68 KB |
| `polymer_mpk_dataset.csv` | Virtual polymer dataset (40971 records) | ~285 MB |
| `pe_solvent_dataset.csv` | PE solvent dataset (9828 records) | ~65 MB |

## Installation

### Requirements

- Python 3.9+
- Required packages listed in `requirements.txt`

### Setup

1. Clone this repository:
```bash
git clone https://github.com/[username]/analysis-polymer-chemical-resistance.git
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

[Contact information will be added]

