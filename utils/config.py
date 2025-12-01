#!/usr/bin/env python3
"""
Configuration constants for chemical resistance analysis.

This module contains all configuration values, paths, parameters, and constants
used throughout the analysis pipeline. Centralizing these values makes it easy
to modify settings and maintain consistency across the codebase.
"""

from pathlib import Path
import sys

# =============================================================================
# DIRECTORY PATHS
# =============================================================================

# Root directory (repository root: analysis-polymer-chemical-resistance/)
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(ROOT_DIR.as_posix())

# Data directory (download from Zenodo)
DATA_DIR = ROOT_DIR / "data"

# Output directories
OUTPUT_DIR = ROOT_DIR / "output"
INTERIM_DIR = OUTPUT_DIR / "interim"
REPORT_DIR_MAIN = OUTPUT_DIR / "Main"
REPORT_DIR_SUB = OUTPUT_DIR / "Sub"
MODEL_DIR = OUTPUT_DIR / "Model"
LOGOCV_DIR = OUTPUT_DIR / "LOGOCV"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
INTERIM_DIR.mkdir(exist_ok=True, parents=True)
REPORT_DIR_MAIN.mkdir(exist_ok=True, parents=True)
REPORT_DIR_SUB.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)
LOGOCV_DIR.mkdir(exist_ok=True, parents=True)

# =============================================================================
# RANDOM SEED
# =============================================================================

SEED = 42

# =============================================================================
# THRESHOLDS
# =============================================================================

# Model prediction threshold
THRESHOLD = 0.9

# Resistance classification thresholds
Th_high = 0.9  # Non-resistant threshold (p > Th_high)
Th_low = 0.1   # Resistant threshold (p < Th_low)

# Standard deviation threshold for feature filtering
STD_THRESHOLD = 1e-10

# =============================================================================
# FONT SIZES FOR PLOTTING
# =============================================================================

BASE_FONT_SIZE = 40
TITLE_FONT_SIZE = int(BASE_FONT_SIZE * 1.25)      # 50
LABEL_FONT_SIZE = int(BASE_FONT_SIZE * 1.125)     # 45
TICK_FONT_SIZE = BASE_FONT_SIZE                   # 40
ANNOTATION_FONT_SIZE = BASE_FONT_SIZE - 2          # 38

# =============================================================================
# COLORS FOR PLOTTING
# =============================================================================

COLOR_RESISTANT = 'green'
COLOR_NON_RESISTANT = 'red'
ALPHA_HIST = 0.3
ALPHA_FILL = 0.2

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

# GBDT (Gradient Boosting Decision Tree) parameters
GBDT_PARAMS = {
    "learning_rate": 0.0995950237281464,
    "n_estimators": 499,
    "max_depth": 10,
    "min_samples_split": 9,
    "min_samples_leaf": 6,
    "subsample": 0.738723859213826,
    "max_features": None
}

# Feature selection
FEATURE_NO = 15  # Index in all_features_combinations
EXCLUDE_COLS = ['n_atom_radonpy_polymer', 'mol_weight_radonpy_polymer']

# =============================================================================
# COLUMN NAMES
# =============================================================================

# Target columns
CRYSTALLINITY_BINARY_COL = ["crystalinity_binary"]
CHI_COL = ["chi"]

# Selected RadonPy columns for analysis
RADONPY_CHOICE_POLYMER_COLS = [
    "density",
    # "self-diffusion",
    # "Cp",
    # "Cv",
    # "Mn",
    # "Mw",
    # "mol_weight",
    # "tg_max_temp",
]

# =============================================================================
# POLYMER AND SOLVENT LABELS
# =============================================================================

# Non-crystalline polymers (for special coloring in figures)
NON_CRYSTALLINE_POLYMERS = ['PVA', 'PIB', 'PMMA', 'PVdC', 'PC', 'PVC', 'PS', 'PSF', 'PMP']

# Polymer order for Supplementary Figure S1 (top to bottom)
SYMBOL_ORDER_S1 = [
    'PVA', 'PIB', 'PMMA', 'PVdC', 'PC', 'PVC', 'PS', 'PSF', 
    'PE', 'PP', 'PA12', 'PA', 'PA6', 'PMP', 'PPCO', 'POM', 
    'PET', 'PVDF', 'CP', 'ECTFE', 'ETFE', 'PFA', 'PBT', 'PPS', 
    'PTFE', 'PEEK', 'FEP'
]

# Practical polymer columns for CV evaluation
PRACTICAL_POLYMER_SYMBOLS = ["PTFE", "PE", "PVC"]

# =============================================================================
# FEATURE COMBINATION LABELS
# =============================================================================

LOG_STR_DICT = {
    0: "FFwoPol",
    1: "FF",
    2: "FFwoPol_CL",
    3: "FF_CL",
    4: "FFwoPol_RP",
    5: "FF_RP",
    6: "FFwoPol_CL_RP",
    7: "FF_CL_RP",
    8: "FFwoPol_Chi",
    9: "FF_Chi",
    10: "FFwoPol_CL_Chi",
    11: "FF_CL_Chi",
    12: "FFwoPol_RP_Chi",
    13: "FF_RP_Chi",
    14: "FFwoPol_CL_RP_Chi",
    15: "FF_CL_RP_Chi",
}

# =============================================================================
# FEATURE COLUMN EXTRACTION FUNCTIONS
# =============================================================================

def get_feature_columns(valid_data):
    """
    Extract feature column names from valid_data DataFrame.
    
    This function categorizes columns based on their naming conventions:
    - FF_*_solvent: Force field parameters for solvents
    - FF_*_polymer: Force field parameters for polymers
    - RDKit_*_solvent: RDKit descriptors for solvents
    - RDKit_*_polymer: RDKit descriptors for polymers
    
    Parameters
    ----------
    valid_data : pd.DataFrame
        DataFrame containing feature columns
    
    Returns
    -------
    dict
        Dictionary containing lists of column names for each category
    """
    # Force field columns
    FF_solvent_cols = [
        col for col in valid_data if col.endswith("_solvent") and col.startswith("FF_")
    ]
    FF_polymer_cols = [
        col for col in valid_data if col.endswith("_polymer") and col.startswith("FF_")
    ]
    
    # RDKit descriptor columns
    RDKit_solvent_cols = [
        col for col in valid_data if col.endswith("_solvent") and col.startswith("RDKit_")
    ]
    RDKit_polymer_cols = [
        col for col in valid_data if col.endswith("_polymer") and col.startswith("RDKit_")
    ]
    
    # Force field parameter categories (solvent)
    FF_mass_solvent_cols = [
        col for col in valid_data if col.endswith("_solvent") and col.startswith("FF_mass")
    ]
    FF_charge_solvent_cols = [
        col for col in valid_data if col.endswith("_solvent") and col.startswith("FF_charge")
    ]
    FF_epsilon_solvent_cols = [
        col for col in valid_data if col.endswith("_solvent") and col.startswith("FF_epsilon")
    ]
    FF_sigma_solvent_cols = [
        col for col in valid_data if col.endswith("_solvent") and col.startswith("FF_sigma")
    ]
    FF_k_bond_solvent_cols = [
        col for col in valid_data if col.endswith("_solvent") and col.startswith("FF_k_bond")
    ]
    FF_r0_solvent_cols = [
        col for col in valid_data if col.endswith("_solvent") and col.startswith("FF_r0")
    ]
    FF_polar_solvent_cols = [
        col for col in valid_data if col.endswith("_solvent") and col.startswith("FF_polar")
    ]
    FF_k_angle_solvent_cols = [
        col for col in valid_data if col.endswith("_solvent") and col.startswith("FF_k_angle")
    ]
    FF_theta0_solvent_cols = [
        col for col in valid_data if col.endswith("_solvent") and col.startswith("FF_theta0")
    ]
    FF_k_dih_solvent_cols = [
        col for col in valid_data if col.endswith("_solvent") and col.startswith("FF_k_dih")
    ]
    
    # Force field parameter categories (polymer)
    FF_mass_polymer_cols = [
        col for col in valid_data if col.endswith("_polymer") and col.startswith("FF_mass")
    ]
    FF_charge_polymer_cols = [
        col for col in valid_data if col.endswith("_polymer") and col.startswith("FF_charge")
    ]
    FF_epsilon_polymer_cols = [
        col for col in valid_data if col.endswith("_polymer") and col.startswith("FF_epsilon")
    ]
    FF_sigma_polymer_cols = [
        col for col in valid_data if col.endswith("_polymer") and col.startswith("FF_sigma")
    ]
    FF_k_bond_polymer_cols = [
        col for col in valid_data if col.endswith("_polymer") and col.startswith("FF_k_bond")
    ]
    FF_r0_polymer_cols = [
        col for col in valid_data if col.endswith("_polymer") and col.startswith("FF_r0")
    ]
    FF_polar_polymer_cols = [
        col for col in valid_data if col.endswith("_polymer") and col.startswith("FF_polar")
    ]
    FF_k_angle_polymer_cols = [
        col for col in valid_data if col.endswith("_polymer") and col.startswith("FF_k_angle")
    ]
    FF_theta0_polymer_cols = [
        col for col in valid_data if col.endswith("_polymer") and col.startswith("FF_theta0")
    ]
    FF_k_dih_polymer_cols = [
        col for col in valid_data if col.endswith("_polymer") and col.startswith("FF_k_dih")
    ]
    
    # Combined force field parameter categories
    FF_mass_cols = [col for col in valid_data if col.startswith("FF_mass")]
    FF_charge_cols = [col for col in valid_data if col.startswith("FF_charge")]
    FF_epsilon_cols = [col for col in valid_data if col.startswith("FF_epsilon")]
    FF_sigma_cols = [col for col in valid_data if col.startswith("FF_sigma")]
    FF_k_bond_cols = [col for col in valid_data if col.startswith("FF_k_bond")]
    FF_r0_cols = [col for col in valid_data if col.startswith("FF_r0")]
    FF_polar_cols = [col for col in valid_data if col.startswith("FF_polar")]
    FF_k_angle_cols = [col for col in valid_data if col.startswith("FF_k_angle")]
    FF_theta0_cols = [col for col in valid_data if col.startswith("FF_theta0")]
    FF_k_dih_cols = [col for col in valid_data if col.startswith("FF_k_dih")]
    
    # Force field without polar columns
    FF_solvent_wo_polars_cols = [
        col for col in FF_solvent_cols if not col.startswith("FF_polar")
    ]
    
    return {
        'FF_solvent_cols': FF_solvent_cols,
        'FF_polymer_cols': FF_polymer_cols,
        'RDKit_solvent_cols': RDKit_solvent_cols,
        'RDKit_polymer_cols': RDKit_polymer_cols,
        'FF_mass_solvent_cols': FF_mass_solvent_cols,
        'FF_charge_solvent_cols': FF_charge_solvent_cols,
        'FF_epsilon_solvent_cols': FF_epsilon_solvent_cols,
        'FF_sigma_solvent_cols': FF_sigma_solvent_cols,
        'FF_k_bond_solvent_cols': FF_k_bond_solvent_cols,
        'FF_r0_solvent_cols': FF_r0_solvent_cols,
        'FF_polar_solvent_cols': FF_polar_solvent_cols,
        'FF_k_angle_solvent_cols': FF_k_angle_solvent_cols,
        'FF_theta0_solvent_cols': FF_theta0_solvent_cols,
        'FF_k_dih_solvent_cols': FF_k_dih_solvent_cols,
        'FF_mass_polymer_cols': FF_mass_polymer_cols,
        'FF_charge_polymer_cols': FF_charge_polymer_cols,
        'FF_epsilon_polymer_cols': FF_epsilon_polymer_cols,
        'FF_sigma_polymer_cols': FF_sigma_polymer_cols,
        'FF_k_bond_polymer_cols': FF_k_bond_polymer_cols,
        'FF_r0_polymer_cols': FF_r0_polymer_cols,
        'FF_polar_polymer_cols': FF_polar_polymer_cols,
        'FF_k_angle_polymer_cols': FF_k_angle_polymer_cols,
        'FF_theta0_polymer_cols': FF_theta0_polymer_cols,
        'FF_k_dih_polymer_cols': FF_k_dih_polymer_cols,
        'FF_mass_cols': FF_mass_cols,
        'FF_charge_cols': FF_charge_cols,
        'FF_epsilon_cols': FF_epsilon_cols,
        'FF_sigma_cols': FF_sigma_cols,
        'FF_k_bond_cols': FF_k_bond_cols,
        'FF_r0_cols': FF_r0_cols,
        'FF_polar_cols': FF_polar_cols,
        'FF_k_angle_cols': FF_k_angle_cols,
        'FF_theta0_cols': FF_theta0_cols,
        'FF_k_dih_cols': FF_k_dih_cols,
        'FF_solvent_wo_polars_cols': FF_solvent_wo_polars_cols,
    }


def get_all_features_combinations(FF_solvent_cols, FF_polymer_cols, FF_solvent_wo_polars_cols,
                                   crystalinity_binary_col, radonpy_polymer_cols, chi_col):
    """
    Generate all feature combinations for model training.
    
    Parameters
    ----------
    FF_solvent_cols : list
        Force field columns for solvents
    FF_polymer_cols : list
        Force field columns for polymers
    FF_solvent_wo_polars_cols : list
        Force field columns for solvents without polar parameters
    crystalinity_binary_col : list
        Crystallinity binary column
    radonpy_polymer_cols : list
        RadonPy columns for polymers
    chi_col : list
        Chi parameter column
    
    Returns
    -------
    list
        List of feature combinations
    """
    return [
        FF_solvent_wo_polars_cols + FF_polymer_cols,  # 0
        FF_solvent_cols + FF_polymer_cols,  # 1
        FF_solvent_wo_polars_cols + FF_polymer_cols + crystalinity_binary_col,  # 2
        FF_solvent_cols + FF_polymer_cols + crystalinity_binary_col,  # 3
        FF_solvent_wo_polars_cols + FF_polymer_cols + radonpy_polymer_cols,  # 4
        FF_solvent_cols + FF_polymer_cols + radonpy_polymer_cols,  # 5
        FF_solvent_wo_polars_cols + FF_polymer_cols + crystalinity_binary_col + radonpy_polymer_cols,  # 6
        FF_solvent_cols + FF_polymer_cols + crystalinity_binary_col + radonpy_polymer_cols,  # 7
        FF_solvent_wo_polars_cols + FF_polymer_cols + chi_col,  # 8
        FF_solvent_cols + FF_polymer_cols + chi_col,  # 9
        FF_solvent_wo_polars_cols + FF_polymer_cols + crystalinity_binary_col + chi_col,  # 10
        FF_solvent_cols + FF_polymer_cols + crystalinity_binary_col + chi_col,  # 11
        FF_solvent_wo_polars_cols + FF_polymer_cols + radonpy_polymer_cols + chi_col,  # 12
        FF_solvent_cols + FF_polymer_cols + radonpy_polymer_cols + chi_col,  # 13
        FF_solvent_wo_polars_cols + FF_polymer_cols + crystalinity_binary_col + radonpy_polymer_cols + chi_col,  # 14
        FF_solvent_cols + FF_polymer_cols + crystalinity_binary_col + radonpy_polymer_cols + chi_col,  # 15
    ]


def get_cv_column_names(train_info):
    """
    Generate column names for cross-validation based on training info.
    
    Parameters
    ----------
    train_info : pd.DataFrame
        Training information DataFrame with 'symbol' and 'cluster_labels' columns
    
    Returns
    -------
    dict
        Dictionary containing:
        - all_polymer_cols: CV column names for all polymers
        - solvent_cols: CV column names for solvent clusters
        - practical_polymer_cols: CV column names for practical polymers
    """
    all_polymer_cols = [
        "cv_" + symbol 
        for symbol in train_info.drop_duplicates(subset="symbol")["symbol"]
    ]
    solvent_cols = [
        "cv_solv_cluster_" + str(label) 
        for label in train_info.drop_duplicates(subset="cluster_labels")["cluster_labels"]
    ]
    practical_polymer_cols = [
        "cv_" + symbol for symbol in PRACTICAL_POLYMER_SYMBOLS
    ]
    
    return {
        'all_polymer_cols': all_polymer_cols,
        'solvent_cols': solvent_cols,
        'practical_polymer_cols': practical_polymer_cols
    }


def create_logocv_directory(classification_type: str = "2_class") -> str:
    """
    Create LOGOCV directory for saving validation results.
    Uses a fixed directory name for reproducibility.
    
    Parameters
    ----------
    classification_type : str
        Classification type string (e.g., "2_class", "4_class")
    
    Returns
    -------
    str
        Path to created directory
    """
    import os
    dir_name = LOGOCV_DIR
    os.makedirs(dir_name, exist_ok=True)
    return str(dir_name)


def get_features_for_model(all_features_combinations, feature_no=FEATURE_NO, exclude_cols=None):
    """
    Get filtered feature list for model training.
    
    Parameters
    ----------
    all_features_combinations : list
        List of all feature combinations
    feature_no : int
        Index of feature combination to use
    exclude_cols : list, optional
        Columns to exclude from features
    
    Returns
    -------
    list
        Filtered list of feature column names
    """
    if exclude_cols is None:
        exclude_cols = EXCLUDE_COLS
    
    features = list(all_features_combinations[feature_no])
    features = [col for col in features if col not in exclude_cols]
    
    return features

