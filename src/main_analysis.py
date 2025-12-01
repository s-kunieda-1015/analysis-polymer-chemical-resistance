#!/usr/bin/env python
# coding: utf-8
"""
Chemical Resistance Prediction Analysis for Paper Publication

This script generates all figures and tables for the paper on chemical resistance
prediction using machine learning models. It includes:
- Main figures (Fig1, Fig3-9)
- Supplementary figures (S1-S6)
- Model training and evaluation (optional, controlled by flags)
- Dataset export for publication
"""
from __future__ import annotations

# EXECUTION FLAGS (set via command line arguments)
import argparse
parser = argparse.ArgumentParser(description='Chemical Resistance Prediction Analysis')
parser.add_argument('--skip-resistance-training', action='store_true', 
                    help='Skip resistance prediction model training (default: train)')
parser.add_argument('--skip-chi-training', action='store_true',
                    help='Skip chi parameter prediction model training (default: train)')
parser.add_argument('--skip-crystallinity-training', action='store_true',
                    help='Skip crystallinity model training (default: train with 9-fold CV 100 iterations)')
parser.add_argument('--skip-logocv', action='store_true',
                    help='Skip LOGOCV validation (time-consuming)')
parser.add_argument('--skip-mic', action='store_true',
                    help='Skip MIC calculation (time-consuming)')
parser.add_argument('--n-jobs', type=int, default=8,
                    help='Max number of parallel jobs for LOGOCV (default: 8, auto-adjusted by group count)')
args = parser.parse_args()

TRAIN_RESISTANCE_MODEL = not args.skip_resistance_training  # Chemical resistance prediction model (GBDT) - DEFAULT: True
TRAIN_CHI_MODEL = not args.skip_chi_training                # Chi parameter prediction model (XGBoost) - DEFAULT: True
TRAIN_CRYSTALLINITY_MODEL = not args.skip_crystallinity_training  # Crystallinity classification model (RandomForest) - DEFAULT: True
SKIP_LOGOCV = args.skip_logocv                  # Skip LOGOCV validation
SKIP_MIC = args.skip_mic                        # Skip MIC calculation
N_JOBS = args.n_jobs                            # Number of parallel jobs for LOGOCV

# SECTION 1: IMPORTS
import os
import sys
import json
import csv
import re
import math
import argparse
from pathlib import Path
from datetime import datetime
from itertools import chain, combinations
from typing import TYPE_CHECKING, Any, Sequence, List, Optional, TypedDict

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.ticker as mticker
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.transforms import blended_transform_factory
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch
import seaborn as sns
import japanize_matplotlib  # Used for rendering Japanese characters

from sklearn.model_selection import KFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support,
    roc_curve, auc, r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from joblib import dump, load, Parallel, delayed
import joblib
from minepy import MINE
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Draw

# Add parent directory to path for utils imports (when running from src/)
import sys
from pathlib import Path
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
sys.path.insert(0, str(_repo_root))

from utils.io import save_as_csv, save_as_csv_with_head
from utils.visualization import init_visualization

# Import utility modules
from utils.config import *
from utils import plotting
from utils import validation
from utils import train
from utils import data_preprocessing
from utils import metrics
from utils import mic

# SECTION 3: DATA LOADING
print("Loading preprocessed datasets...")

# Load chemical resistance dataset
_cr_dataset = data_preprocessing.load_chemical_resistance_dataset()
valid_info_cluster = _cr_dataset.valid_info_cluster
valid_data_cluster = _cr_dataset.valid_data_cluster
feature_cols_dict = _cr_dataset.feature_cols_dict
polymer_name_dict = _cr_dataset.polymer_name_dict
solvent_name_dict = _cr_dataset.solvent_name_dict
radonpy_polymer_cols = _cr_dataset.radonpy_polymer_cols

# Unpack feature column lists
FF_solvent_cols = feature_cols_dict['FF_solvent_cols']
FF_polymer_cols = feature_cols_dict['FF_polymer_cols']
RDKit_solvent_cols = feature_cols_dict['RDKit_solvent_cols']
RDKit_polymer_cols = feature_cols_dict['RDKit_polymer_cols']
FF_solvent_wo_polars_cols = feature_cols_dict['FF_solvent_wo_polars_cols']

# Also unpack detailed FF parameter columns for later use
FF_mass_solvent_cols = feature_cols_dict['FF_mass_solvent_cols']
FF_charge_solvent_cols = feature_cols_dict['FF_charge_solvent_cols']
FF_epsilon_solvent_cols = feature_cols_dict['FF_epsilon_solvent_cols']
FF_sigma_solvent_cols = feature_cols_dict['FF_sigma_solvent_cols']
FF_k_bond_solvent_cols = feature_cols_dict['FF_k_bond_solvent_cols']
FF_r0_solvent_cols = feature_cols_dict['FF_r0_solvent_cols']
FF_polar_solvent_cols = feature_cols_dict['FF_polar_solvent_cols']
FF_k_angle_solvent_cols = feature_cols_dict['FF_k_angle_solvent_cols']
FF_theta0_solvent_cols = feature_cols_dict['FF_theta0_solvent_cols']
FF_k_dih_solvent_cols = feature_cols_dict['FF_k_dih_solvent_cols']

FF_mass_polymer_cols = feature_cols_dict['FF_mass_polymer_cols']
FF_charge_polymer_cols = feature_cols_dict['FF_charge_polymer_cols']
FF_epsilon_polymer_cols = feature_cols_dict['FF_epsilon_polymer_cols']
FF_sigma_polymer_cols = feature_cols_dict['FF_sigma_polymer_cols']
FF_k_bond_polymer_cols = feature_cols_dict['FF_k_bond_polymer_cols']
FF_r0_polymer_cols = feature_cols_dict['FF_r0_polymer_cols']
FF_polar_polymer_cols = feature_cols_dict['FF_polar_polymer_cols']
FF_k_angle_polymer_cols = feature_cols_dict['FF_k_angle_polymer_cols']
FF_theta0_polymer_cols = feature_cols_dict['FF_theta0_polymer_cols']
FF_k_dih_polymer_cols = feature_cols_dict['FF_k_dih_polymer_cols']

FF_mass_cols = feature_cols_dict['FF_mass_cols']
FF_charge_cols = feature_cols_dict['FF_charge_cols']
FF_epsilon_cols = feature_cols_dict['FF_epsilon_cols']
FF_sigma_cols = feature_cols_dict['FF_sigma_cols']
FF_k_bond_cols = feature_cols_dict['FF_k_bond_cols']
FF_r0_cols = feature_cols_dict['FF_r0_cols']
FF_polar_cols = feature_cols_dict['FF_polar_cols']
FF_k_angle_cols = feature_cols_dict['FF_k_angle_cols']
FF_theta0_cols = feature_cols_dict['FF_theta0_cols']
FF_k_dih_cols = feature_cols_dict['FF_k_dih_cols']

# Set valid_info and valid_data to point to cluster versions
valid_info = valid_info_cluster.copy()
valid_data = valid_data_cluster.copy()

# Initialize visualization settings
from utils.visualization import init_visualization
init_visualization()
plotting.set_paper_style()

# Create df_merged for clustering visualization (needed later)
df_info = valid_info.copy()
df_feats = valid_data.copy()
df_info = df_info[~df_info.index.duplicated(keep='first')]
df_feats = df_feats[~df_feats.index.duplicated(keep='first')]
df_merged = pd.merge(df_info, df_feats, how="inner", left_index=True, right_index=True)

# Calculate solvent clusters for visualization
percentage_df = (df_merged
                 .groupby('solvent')['resistance_binary']
                 .apply(plotting.calculate_percentage)
                 .unstack()
                 .reset_index()
                )
percentage_df.fillna(0, inplace=True)
merged_df = df_merged.merge(percentage_df, on='solvent', how='left')
columns_rename = {'A': 'A_original', 0: 'resistance_0_(%)', 1: 'resistance_1_(%)'}
merged_df.rename(columns=columns_rename, inplace=True)
df_solv = merged_df.drop_duplicates(subset="solvent").reset_index(drop=True)
df_solv_cluster = df_solv.copy()
df_solv_cluster.sort_values("resistance_0_(%)", ascending=True, inplace=True)
df_solv_cluster.reset_index(drop=True, inplace=True)
df_solv_cluster['rank'] = df_solv_cluster['resistance_0_(%)'].rank(method='min')
clusters = np.array_split(df_solv_cluster, 10)
for i, cluster in enumerate(clusters):
    df_solv_cluster.loc[cluster.index, 'cluster_labels'] = i+1
df_solv_cluster['cluster_labels'] = df_solv_cluster['cluster_labels'].astype(int)
average_df = df_solv_cluster.groupby('cluster_labels')[['resistance_0_(%)', 'resistance_1_(%)']].mean()
average_df.reset_index(inplace=True)

# Note: Solvent renaming data is now included in the metadata JSON loaded by data_preprocessing
# df_rename_solvent is no longer needed as a separate file

# Try to load polymer_mpk and pe_solvent datasets if available
try:
    _df_polymer_mpk = data_preprocessing.load_polymer_mpk_dataset()
    _df_pe_solvent = data_preprocessing.load_pe_solvent_dataset()
    _POLYMER_MPK_LOADED = True
    _PE_SOLVENT_LOADED = True
    print(f"✓ polymer_mpk_dataset loaded: {len(_df_polymer_mpk)} records")
    print(f"✓ pe_solvent_dataset loaded: {len(_df_pe_solvent)} records")
except FileNotFoundError:
    _POLYMER_MPK_LOADED = False
    _PE_SOLVENT_LOADED = False
    print("Note: polymer_mpk and pe_solvent datasets not found in processed dir.")
    print("      These will be loaded from original files.")

print(f"✓ Preprocessed data loaded: {len(valid_info_cluster)} records")

# Count the binary resistance indicator values
resistance_counts = valid_info_cluster['resistance_binary'].value_counts()

# Rename the index labels: map 0 to "OK" and 1 to "NG"
resistance_labels = {0: 'OK', 1: 'NG'}
resistance_counts = resistance_counts.rename(index=resistance_labels)

# Specify the order for the bar chart (OK first, then NG)
resistance_order = ['OK', 'NG']

fig, ax = plt.subplots(figsize=(36, 24))

# Draw the bar chart
bars = ax.bar(resistance_order, [resistance_counts.get(label, 0) for label in resistance_order])

# Format the y-axis with commas and set labels and title
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.ylabel('Data Count', fontsize=120, labelpad=24)
# Set the font size for y-axis tick labels
ax.tick_params(axis='y', labelsize=90)
ax.set_ylim(0, 2500)
plt.title(f'Distribution of Binary Resistance Indicators ({resistance_counts.sum():,} Data Points)', fontsize=80, y=1.02)

# Set custom x-axis tick labels with corresponding codes
ax.set_xticklabels(['0 : OK', '1 : NG'], fontsize=120)

# Add numerical labels above each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:,}', ha='center', va='bottom', fontsize=120)

# Obtain the current datetime for generating a unique filename
now = datetime.now()
filename = now.strftime('%Y%m%d%H%M%S')

plt.tight_layout()

# Resistance data count distribution by polymer (binary classification)
fig, ax, _, _ = plotting._plot_resistance_binary_dists(valid_info_cluster, polymer_name_dict, "smiles_polymer",is_reverse=True,is_rotate=True)
ax.set(xlabel="resistance count", ylabel="polymer")

# Resistance data rate distribution by polymer (binary classification)
fig, ax, polymer_resistance_rank, _ = plotting._plot_resistance_binary_dists(valid_info_cluster, polymer_name_dict, "smiles_polymer", is_rate=True,is_reverse=True,is_rotate=True)
ax.set(xlabel="resistance rate", ylabel="polymer")

# Resistance data rate distribution by polymer (binary classification)
fig, ax, polymer_resistance_rank, _ = plotting._plot_resistance_binary_dists(valid_info_cluster, polymer_name_dict, "smiles_polymer", is_rate=True, is_reverse=True)
ax.set(xlabel="resistance rate", ylabel="polymer")

# Resistance data count distribution by solvent (binary classification)
fig, ax, _, _ = plotting._plot_resistance_binary_dists(valid_info_cluster, solvent_name_dict, "smiles_solvent",is_reverse=True)
ax.set(xlabel="resistance count", ylabel="solvent")

# Resistance data count distribution by solvent (binary classification)
fig, ax, solvent_rank, _ = plotting._plot_resistance_binary_dists(valid_info_cluster, solvent_name_dict, "smiles_solvent", is_rate=True,is_reverse=True)
ax.set(xlabel="resistance count", ylabel="solvent")

# # Output per solvent cluster

# ----- Case: is_rate=True -----
results_rate = plotting.plot_resistance_by_cluster_fixed(
    valid_info_cluster,       # DataFrame with cluster information
    solvent_name_dict,        # Dictionary holding chemical names (e.g., "jp_abbr")
    "smiles_solvent",
    cluster_col="cluster_labels",
    is_rate=True,
    is_reverse=True,
    with_structures=True,     # Place chemical names and structure images outside the graph
    image_zoom=0.25,
    label_x=-0.13,            # x-coordinate for placing chemical names outside the graph (aligned far left)
    image_x_offset=0.05
)

for cluster, (fig, ax) in results_rate.items():
    ax.set(xlabel="resistance ratio", ylabel="")
    fig.tight_layout()

# ----- Case: is_rate=False -----
results_count = plotting.plot_resistance_by_cluster_fixed(
    valid_info_cluster,
    solvent_name_dict,
    "smiles_solvent",
    cluster_col="cluster_labels",
    is_rate=False,            # Plot based on data count
    is_reverse=True,
    with_structures=True,
    image_zoom=0.25,
    label_x=-0.13,
    image_x_offset=0.05
)

for cluster, (fig, ax) in results_count.items():
    ax.set(xlabel="resistance count", ylabel="")
    fig.tight_layout()

# ---------------------------
# Set overall default font size
# ---------------------------
base_fontsize = 14  # Changing this adjusts the overall font size
plt.rcParams.update({'font.size': base_fontsize})

# Here, it is assumed that valid_info_cluster (pd.DataFrame),
# solvent_name_dict (dictionary of chemical names and structure images),
# and REPORT_DIR_SUB (save destination directory) are already defined.

# Example: The column name for cluster labels is "cluster_labels", and the key for structural formulas is "smiles_solvent".
cluster_col = "cluster_labels"
key_col = "smiles_solvent"

# Mode to use (e.g., case is_rate=True)
is_rate = True  
is_reverse = True
is_rotate = False
with_structures = True

# List of all clusters (numeric or string cluster labels)
clusters = sorted(valid_info_cluster[cluster_col].unique())
n_clusters = len(clusters)

# Get 2 per group to summarize as 2 rows 1 column subplots (handles case where last one is single)
grouped_clusters = [clusters[i:i+2] for i in range(0, n_clusters, 2)]

# Save file number counter
file_no = 1

# Draw as one image file for each group
for group in grouped_clusters:
    # Adjust overall figure size (height for 2 rows. Change as needed)
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(9.6, 12))
    
    # Convert to list if there is only one subplot
    if len(group) == 1:
        axs = [axs]
    
    # Draw for each cluster in the group
    for ax, cluster in zip(axs, group):
        # Extract data for the corresponding cluster
        cluster_data = valid_info_cluster[valid_info_cluster[cluster_col] == cluster]
        
        # Draw bar graphs using internal drawing function
        # _plot_resistance_binary_dists_fixed is designed to accept ax
        ticklabels, keys = plotting._plot_resistance_binary_dists_fixed(
            cluster_data,
            solvent_name_dict,
            key_col,
            ax,
            sort_key=[0],   # Adjust sort_key as needed
            is_rate=is_rate,
            is_reverse=is_reverse,
            is_rotate=is_rotate,
        )
        
        # Title settings
        if not is_rate:
            cluster_count = cluster_data.shape[0]
            ax.set_title(f"Cluster {cluster} (n = {cluster_count})", fontsize=base_fontsize)
        else:
            ax.set_title(f"Cluster {cluster}", fontsize=base_fontsize)
        
        # Set axis labels using base_fontsize (fine-tuning possible if needed)
        if is_rate:
            ax.set(xlabel="resistance ratio", ylabel="")
            # Set axis range to 0-1 for rates
            if is_rotate:
                ax.set_ylim(0, 1)
            else:
                ax.set_xlim(0, 1)
        else:
            ax.set(xlabel="resistance count", ylabel="")
        
        # Add structure images outside the graph (left side)
        if with_structures:
            plotting.add_structure_images(
                ax,
                keys,
                solvent_name_dict,
                plotting.get_rdkit_image,
                image_zoom=0.25,    # Change according to usage
                label_x=-0.13,      # Reference position (appearance can be adjusted according to fontsize)
                image_x_offset=0.05,
                image_y_offset=0.0
            )
    
    fig.tight_layout(pad=3.0)
    file_no += 1

# # Polymer Distribution

# =============================================================================
# SUPPLEMENTARY FIGURE S1: Chemical Structure Images
# =============================================================================
# # S1

# --- Global Settings ---
plt.rcParams.update({
    'font.size': 12,         # Base font size
    'axes.titlesize': 16,      # Title font size
    'axes.labelsize': 14,      # Axis label font size
    'xtick.labelsize': 10,     # x-axis tick label font size
    'ytick.labelsize': 10,     # y-axis tick label font size
})

# Note: plotting functions are now imported from utils.plotting module
# Duplicate definitions removed for code clarity

# Note: Duplicate definition of _plot_resistance_binary_dists removed (originally at line 1496)
# The related usage code below also removed as it was part of the duplicate section

# # Solvent Clusters

# Calculate average indicator ratio for solvent clusters
# Group by 'cluster_labels' and calculate mean of 'resistance_0_(%)' and 'resistance_1_(%)'
average_df = df_solv_cluster.groupby('cluster_labels')[['resistance_0_(%)', 'resistance_1_(%)']].mean()

average_df.reset_index(inplace=True)
average_df

# # Output together

# # Solvent and Resin Resistance Map

polymer_resistance_rank

# Convert solvent_rank index to dictionary
solvent_to_rank = {solvent: idx for idx, solvent in enumerate(solvent_rank)}

# Get rank corresponding to solvent column and add as new column
valid_info_cluster['solvent_rank'] = valid_info_cluster['solvent'].map(solvent_to_rank)

# Extract symbols and create the polymer_to_rank dictionary
polymer_to_rank = {polymer: idx for idx, polymer in enumerate(polymer_resistance_rank)}

# Get rank corresponding to symbol column and add as new column
valid_info_cluster['polymer_rank'] = valid_info_cluster['symbol'].map(polymer_to_rank)

# Check results
valid_info_cluster.head()

# =============================================================================
# SUPPLEMENTARY FIGURE S1: Resistance Binary Distribution by Polymer
# =============================================================================
# Generate two versions: rate (ratio) and count (absolute numbers) with structure images
# Following the same approach as Figure 3 for polymer name coloring and ordering

# Use constants from config for non-crystalline polymers and symbol order
fig_s1_rate, fig_s1_count = plotting.generate_supplementary_s1_resistance_by_polymer(
    valid_info_cluster=valid_info_cluster,
    polymer_name_dict=polymer_name_dict,
    special_labels=NON_CRYSTALLINE_POLYMERS,
    symbol_order=SYMBOL_ORDER_S1,
    save_dir=REPORT_DIR_SUB,
    image_zoom=0.25
)
plt.show()

# =============================================================================
# FIGURE 1: Chemical Resistance Heatmap (Main Figure)
# =============================================================================
# This section generates Figure 1 for the paper showing the chemical resistance
# matrix of polymer-solvent combinations

# Generate Figure 1: Chemical Resistance Heatmap
df_sorted = valid_info_cluster.sort_values(['solvent_rank', 'polymer_rank'])
fig1 = plotting.generate_figure1_heatmap(valid_info_cluster, save_dir=REPORT_DIR_MAIN)
plt.show()

# =============================================================================
# SUPPLEMENTARY FIGURE S2: Cluster Heatmaps
# =============================================================================
# Generate cluster heatmaps showing resistance patterns for each solvent cluster

plotting.generate_supplementary_s2_cluster_heatmaps(
    valid_info_cluster=valid_info_cluster,
    df_sorted=df_sorted,
    solvent_name_dict=solvent_name_dict,
    save_dir=REPORT_DIR_SUB
)
plt.show()

df_solvent_rename = df_sorted.drop_duplicates(subset="solvent")[["solvent","smiles_solvent"]]
df_solvent_rename.to_excel(os.path.join(OUTPUT_DIR, "solvent_rename.xlsx"))

#  ## 4. Dataset Splitting

#  ### 4.1 Test Set Splitting
# 
# The following splitting patterns are available.
# 
# 1. Split by specifying solvents for test data
# 2. Split by specifying polymers for test data
# 3. Random split so that neither solvent nor polymer has the same SMILES between train/test
# 4. Random split so that no same solvent SMILES between train/test
# 5. Random split so that no same polymer SMILES between train/test
# 6. Split by specifying solvent clusters for test data
# 
# Execute the necessary cell according to the use case, and data will be stored in the following variables.
# 
# - `train_data`: train data
# - `test_data`: test data
# 
#  If `train_data` and `test_data` satisfy the above conditions, you can customize the splitting method.
# 
# Also, cells after 1-6 must be executed for any pattern.

# Preparation for splitting
unique_solvents: list[str] = valid_info["smiles_solvent"].unique().tolist()
unique_polymers: list[str] = valid_info["smiles_polymer"].unique().tolist()
np.random.default_rng(SEED).shuffle(unique_solvents)
np.random.default_rng(SEED).shuffle(unique_polymers)

# # If you want to classify by specifying features

valid_data_cluster = valid_data_cluster[~valid_data_cluster.index.duplicated(keep='first')]
train_data = valid_data_cluster.copy()

test_data = pd.DataFrame(columns=valid_data.columns)

features = FF_solvent_cols+FF_polymer_cols

train_data = train_data[features]
test_data = test_data[features]

# Save train data
save_as_csv(train_data, directory=INTERIM_DIR)
train_info = valid_info_cluster.reindex(train_data.index)
save_as_csv(train_info, directory=INTERIM_DIR)

# Check crystallinity distribution in training data
crystallinity_counts = train_info.drop_duplicates(subset="name_polymer").value_counts("crystalinity_binary")
print(f"Crystallinity distribution in training set:")
print(crystallinity_counts)
print(f"Total training records: {len(train_info)}")

# Save test data (currently empty for this analysis)
save_as_csv(test_data, directory=INTERIM_DIR)
test_info = valid_info.reindex(test_data.index)
save_as_csv(test_info, directory=INTERIM_DIR)

print(f"Test set size: {len(test_info)} records")

#  ### 4.2 Validation Set Splitting
# 
# The following splitting patterns are available.
# 
# 1. Random split
# 2. Leave-one-out split by polymer
# 3. Leave-one-out split by solvent cluster
# 
# Similar to test set splitting, select and execute necessary cells according to use case.
#  The split results are stored in the `folds` variable.

folds: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]

#  ## 5. Model Training
# 
#  ### 5.1 Preparation for Training

# Note: ML tracking functionality removed for simplicity

# # Consideration Point 3: 4-class classification or 2-class classification

# Specify target variable column (binary classification)
y_all = valid_info["resistance_binary"]
y_train = train_info["resistance_binary"]
y_test = test_info["resistance_binary"]

# Note: Data tracking functionality removed for simplicity

#  ### 5.2 Execution of Training
# 
# The applied model can be replaced by rewriting the code to satisfy the arguments and return values of the train function.

# # Consideration Point 4: Classification Algorithm

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin

class Metrics(TypedDict):
    confusion_matrix: npt.NDArray[np.int64]
    fig: ConfusionMatrixDisplay
    classwise_f1: npt.NDArray[np.float64]
    micro_f1: float
    macro_f1: float

def _train_cv_model(x_train: pd.DataFrame, y_train: pd.Series[int]) -> ClassifierMixin:
    """Train GradientBoostingClassifier with optimized hyperparameters."""
    from utils import train as train_module
    return train_module.train_resistance_model(x_train, y_train, params)

all_polymer_cols = ["cv_" + symbol for symbol in train_info.drop_duplicates(subset="symbol")["symbol"]]
solvent_cols = ["cv_solv_cluster_" + str(label) for label in train_info.drop_duplicates(subset="cluster_labels")["cluster_labels"]]

practical_polymer_cols = ["cv_PTFE","cv_PE","cv_PVC"]

# Specify features
all_features_combinations = [FF_solvent_cols+FF_polymer_cols]

# Crystallinity features
crystalinity_binary_col = ["crystalinity_binary"]
all_features_combinations = [FF_solvent_cols+FF_polymer_cols+crystalinity_binary_col]

# Add RadonPy features
radonpy_choice_polymer_cols = [
    "density",
    #"self-diffusion",
    #"Cp",
    #"Cv",
    #"Mn",
    #"Mw",
    #"mol_weight",
    #"tg_max_temp",
]
radonpy_choice_polymer_cols = [col + "_radonpy_polymer" for col in radonpy_choice_polymer_cols]
chi_col = ["chi"]

all_features_combinations = [
    FF_solvent_wo_polars_cols + FF_polymer_cols, #0
    FF_solvent_cols + FF_polymer_cols, #1
    FF_solvent_wo_polars_cols + FF_polymer_cols + crystalinity_binary_col, #2
    FF_solvent_cols + FF_polymer_cols + crystalinity_binary_col, #3
    FF_solvent_wo_polars_cols + FF_polymer_cols + radonpy_polymer_cols, #4
    FF_solvent_cols + FF_polymer_cols + radonpy_polymer_cols, #5
    FF_solvent_wo_polars_cols + FF_polymer_cols + crystalinity_binary_col+ radonpy_polymer_cols, #6
    FF_solvent_cols + FF_polymer_cols + crystalinity_binary_col+ radonpy_polymer_cols, #7
    FF_solvent_wo_polars_cols + FF_polymer_cols + chi_col, #8
    FF_solvent_cols + FF_polymer_cols + chi_col, #9
    FF_solvent_wo_polars_cols + FF_polymer_cols + crystalinity_binary_col + chi_col, #10
    FF_solvent_cols + FF_polymer_cols + crystalinity_binary_col + chi_col, #11
    FF_solvent_wo_polars_cols + FF_polymer_cols + radonpy_polymer_cols + chi_col, #12
    FF_solvent_cols + FF_polymer_cols + radonpy_polymer_cols + chi_col, #13
    FF_solvent_wo_polars_cols + FF_polymer_cols + crystalinity_binary_col+ radonpy_polymer_cols+ chi_col, #14
    FF_solvent_cols + FF_polymer_cols + crystalinity_binary_col+ radonpy_polymer_cols+ chi_col, #15
]

log_str_dict = {
    0:"FFwoPol",
    1:"FF",
    2:"FFwoPol_CL",
    3:"FF_CL",
    4:"FFwoPol_RP",
    5:"FF_RP",
    6:"FFwoPol_CL_RP",
    7:"FF_CL_RP",
    8:"FFwoPol_Chi",
    9:"FF_Chi",
    10:"FFwoPol_CL_Chi",
    11:"FF_CL_Chi",
    12:"FFwoPol_RP_Chi",
    13:"FF_RP_Chi",
    14:"FFwoPol_CL_RP_Chi",
    15:"FF_CL_RP_Chi",
}

with open("radonpy_col_check.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(radonpy_polymer_cols)  # Write in one line

radonpy_polymer_cols

# # Add other indicators

# # Bayesian Optimization 4 (Test Version)

# True for binary classification, False for 4-class classification
binary_classification = True

classification_type, num_classes = metrics.set_classification_type(binary_classification)

# Move cluster_labels to the last column
cluster_labels = valid_info_cluster.pop('cluster_labels')
valid_info_cluster['cluster_labels'] = cluster_labels

# Target feature combination and parameters from config
feature_No = FEATURE_NO
params = GBDT_PARAMS
threshold = THRESHOLD

features = list(all_features_combinations[feature_No])
log_str = log_str_dict[feature_No]
all_features_combinations = [features]

def _train_cv(
    val_cluster: str, x_train: pd.DataFrame, y_train_true: pd.Series,
    x_val: pd.DataFrame, y_val_true: pd.Series, num_classes: int, threshold: float = 0.5
) -> pd.DataFrame:
    """Wrapper function for LOGOCV training."""
    return validation.train_cv_fold(
        val_cluster=val_cluster,
        x_train=x_train,
        y_train_true=y_train_true,
        x_val=x_val,
        y_val_true=y_val_true,
        num_classes=num_classes,
        threshold=threshold,
        params=params,  # Use global params
        dir_name=dir_name  # Use global dir_name
    )

# LOGOCV VALIDATION (can be skipped with --skip-logocv)
if SKIP_LOGOCV:
    print("Skipping LOGOCV validation (use without --skip-logocv to run)")
else:
    print(f"Starting LOGOCV validation with {N_JOBS} parallel jobs (this may take a long time)...")
    
    binary_classification = True    # True for binary classification, False for 4-class classification
    classification_type, num_classes = metrics.set_classification_type(binary_classification)
    y_all, y_train, y_test = metrics.set_y_values(binary_classification, valid_info, train_info, test_info)

    # If there is previously saved data, overwrite with unexecuted feature combinations
    try:
        if remaining_combinations:
            all_features_combinations = remaining_combinations
    except NameError:
        pass

    # Use previous output directory if exists, otherwise create new
    try:
        if prev_dir_name:
            dir_name = prev_dir_name
        else:
            dir_name = create_logocv_directory(classification_type)
    except NameError:
        dir_name = create_logocv_directory(classification_type)

    results = []

    for index, features in tqdm(enumerate(all_features_combinations), total=len(all_features_combinations)):
    
        train_data = valid_data_cluster.copy()[list(features)]
        test_data = pd.DataFrame(columns=valid_data.columns)[list(features)]

        train_info_sel = valid_info_cluster.reindex(train_data.index)
        test_info_sel  = valid_info.reindex(test_data.index)

        dfs = []

        # Calculate optimal n_jobs based on number of groups
        n_polymer_groups = train_info_sel['smiles_polymer'].nunique()
        n_solvent_groups = train_info_sel['cluster_labels'].nunique()
        
        # Use number of groups as n_jobs (or N_JOBS if specified and smaller)
        n_jobs_polymer = min(n_polymer_groups, N_JOBS) if N_JOBS > 0 else n_polymer_groups
        n_jobs_solvent = min(n_solvent_groups, N_JOBS) if N_JOBS > 0 else n_solvent_groups
        
        print(f"  Polymer LOGOCV: {n_polymer_groups} groups, using {n_jobs_polymer} parallel jobs")
        print(f"  Solvent LOGOCV: {n_solvent_groups} groups, using {n_jobs_solvent} parallel jobs")

        # Cross-validation for polymer (parallel processing)
        validation.run_logocv_validation(
            train_data=train_data,
            train_info=train_info_sel,
            y_train=y_train,
            group_column="smiles_polymer",
            cv_prefix="cv",
            dfs=dfs,
            num_classes=num_classes,
            threshold=threshold,
            iloc_index=1,
            n_jobs=n_jobs_polymer,
            train_cv_func=_train_cv,
        )

        # Cross-validation for solvent (parallel processing)
        validation.run_logocv_validation(
            train_data=train_data,
            train_info=train_info_sel,
            y_train=y_train,
            group_column="cluster_labels",
            cv_prefix="cv_solv_cluster",
            dfs=dfs,
            num_classes=num_classes,
            threshold=threshold,
            iloc_index=-1,
            n_jobs=n_jobs_solvent,
            train_cv_func=_train_cv,
        )

        result = pd.concat(dfs, axis=1)
        result['features'] = [features] * len(result)

        metric_names = ["micro_f1", "macro_f1", "auc", "accuracy", "fpr", "fnr"]
        for metric in metric_names:
            polymer_cols_metric = [f"{col}_{metric}" for col in all_polymer_cols]
            practical_polymer_cols_metric = [f"{col}_{metric}" for col in practical_polymer_cols]
            solvent_cols_filtered = [f"{col}_{metric}" for col in solvent_cols]
        
            result[f"all_polymer_mean_{metric}"] = result[polymer_cols_metric].mean(axis=1)
            result[f"all_polymer_std_{metric}"] = result[polymer_cols_metric].std(axis=1)
            result[f"practical_polymer_mean_{metric}"] = result[practical_polymer_cols_metric].mean(axis=1)
            result[f"practical_polymer_std_{metric}"] = result[practical_polymer_cols_metric].std(axis=1)
            result[f"all_solvent_mean_{metric}"] = result[solvent_cols_filtered].mean(axis=1)
            result[f"all_solvent_std_{metric}"] = result[solvent_cols_filtered].std(axis=1)
            result[f"all_mean_{metric}"] = result[[f"all_polymer_mean_{metric}", f"all_solvent_mean_{metric}"]].mean(axis=1)
    
            result = metrics.calc_std_all(result, metric)

        results.append(result)

    # Save final results as CSV
    final_result = pd.concat(results, axis=0)
    for metric in metric_names:
        final_result_metric = final_result.filter(like=f"_{metric}")
        final_result_metric["features"] = final_result["features"]
        final_result_metric.sort_values(f"all_mean_{metric}", ascending=False, inplace=True)
        final_result_metric.reset_index(drop=True, inplace=True)
        filename = os.path.join(dir_name, f"result_{metric}.csv")
        final_result_metric.to_csv(filename)
        print(f"Saved final result for {metric} to {filename}")

# SUPPLEMENTARY FIGURE S3: ROC Curves and Confusion Matrices
# Use LOGOCV_DIR from config
logocv_dir = LOGOCV_DIR
plot_data_dir = logocv_dir / "plot_data"

# Convert symbol_order to list
symbol_order = list(df_sorted.sort_values('polymer_rank')['symbol'].unique())

# Generate S3 figures (saved in reports_paper/Sub/S3/)
plotting.generate_supplementary_s3_roc_confusion(
    plot_data_dir=str(plot_data_dir),
    output_dir=REPORT_DIR_SUB,
    symbol_order=symbol_order,
    font_size=16
)
plt.show()

# Read LOGOCV results
metric_names = ["micro_f1", "macro_f1", "auc", "accuracy", "fpr", "fnr"]
# Use LOGOCV_DIR from config
logocv_dir = LOGOCV_DIR
result_files = {}
for metric in metric_names:
    result_file = logocv_dir / f"result_{metric}.csv"
    if result_file.exists():
        result_files[metric] = result_file
    else:
        print(f"Warning: No result file found for metric '{metric}' in {logocv_dir}")
        result_files[metric] = None

# Read the results for each metric into a DataFrame
dfs = {}
for metric in metric_names:
    if result_files[metric] and result_files[metric].exists():
        dfs[metric] = pd.read_csv(result_files[metric], index_col=0)
        print(f"Loaded {metric} results from {result_files[metric].name}")
    else:
        print(f"Skipping {metric}: file not found")

# Example visualization for one of the metrics (e.g., micro_f1)
df_opt = dfs["micro_f1"]

# Process the DataFrame for visualization
df_polymer = df_opt.iloc[0:1, 0:39]
df_polymer = df_polymer.sort_values(by=df_polymer.index[0], axis=1, ascending=False)

polymer_list = [col.replace('cv_', '') for col in df_polymer.columns]

print(polymer_list)

# Create a new DataFrame with the updated column names
df_new = df_polymer.copy()
df_new.columns = polymer_list

# Display the new DataFrame
print(df_new)

# 4-class classification
def _get_resistance_values(
    info: pd.DataFrame,
    smiles_name_dict: dict[str, dict[str, str]],
    key_col: str,
    *,
    sort_key: Sequence[Any] = [0, 1, 2, 3],
    is_rate: bool = False,
) -> pd.DataFrame:
    key_samples = info.drop_duplicates(subset=key_col)
    keys: list[str] = key_samples[key_col].tolist()
    ticklabels = [smiles_name_dict[key]["jp_abbr"] for key in keys]

    counts = pd.DataFrame(
        {
            ylabel: info.loc[info[key_col] == key, "resistance"].value_counts()
            for key, ylabel in zip(keys, ticklabels, strict=True)
        },
    )
    values = counts.fillna(0).astype(int)
    if is_rate:
        values = values.div(values.sum(axis=0), axis=1)

    values = values.sort_values(by=sort_key, ascending=True, axis=1)

    return values

# 2-class classification
def _get_resistance_binary_values(
    info: pd.DataFrame,
    smiles_name_dict: dict[str, dict[str, str]],
    key_col: str,
    *,
    sort_key: Sequence[Any] = [0, 1],
    is_rate: bool = False,
    is_reverse: bool = False,  # * Added
) -> pd.DataFrame:
    key_samples = info.drop_duplicates(subset=key_col)
    keys: list[str] = key_samples[key_col].tolist()
    
        ticklabels = [smiles_name_dict[key]["jp_abbr"] for key in keys]

    counts = pd.DataFrame(
        {
            ylabel: info.loc[info[key_col] == key, "resistance_binary"].value_counts()
            for key, ylabel in zip(keys, ticklabels, strict=True)
        },
    )

    values = counts.fillna(0).astype(int)
    if is_rate:
        values = values.div(values.sum(axis=0), axis=1)

    # * Order control (supports is_reverse)
    values = values.sort_values(by=sort_key, ascending=not is_reverse, axis=1)

    return values

values = _get_resistance_binary_values(valid_info, polymer_name_dict, "smiles_polymer",is_rate=True,is_reverse=True)

columns = values.columns
new_columns = [re.search('\((.*?)\)', col).group(1) if '(' in col else col for col in columns]
print(new_columns)

# Example visualization for one of the metrics (e.g., micro_f1)
df_opt = dfs["micro_f1"]

# Process the DataFrame for visualization
df_polymer = df_opt.iloc[0:1, 0:39]
df_polymer = df_polymer.sort_values(by=df_polymer.index[0], axis=1, ascending=False)

# Extract the base column names without metrics and 'cv_' prefix
polymer_list = [re.sub(r'^cv_|_(micro_f1|macro_f1|auc|accuracy|fpr|fnr)$', '', col) for col in df_polymer.columns]

# Create a new DataFrame with the updated column names
df_new = df_polymer.copy()
df_new.columns = polymer_list

# Display the new DataFrame
print(df_new)

# Visualization code
def visualize_all_metrics(dfs, values, metric_list=None):
    if metric_list is None:
        metric_list = ["micro_f1", "macro_f1", "auc", "accuracy", "fpr", "fnr"]
    fig, axes = plt.subplots(len(metric_list), 1, figsize=(12, len(metric_list) * 6))

    for ax, metric in zip(axes, metric_list):
        df_opt = dfs[metric]
        df_polymer = df_opt.iloc[0:1, 0:39]
        df_polymer = df_polymer.sort_values(by=df_polymer.index[0], axis=1, ascending=False)

        polymer_list = [re.sub(r'^cv_|_(micro_f1|macro_f1|auc|accuracy|fpr|fnr)$', '', col) for col in df_polymer.columns]
        df_new = df_polymer.copy()
        df_new.columns = polymer_list

        # Ensure values is a DataFrame
        values_df = values.to_frame().T if isinstance(values, pd.Series) else values

        columns = values_df.columns
        new_columns = [re.search(r'\((.*?)\)', col).group(1) if '(' in col else col for col in columns]
        print(new_columns)

        df_new.columns = [col.replace('cv_', '') for col in df_new.columns]
        df_new = df_new[new_columns]
        df_new.columns = columns

        objs = df_new.columns.tolist()
        objs = new_columns

        values = df_new.iloc[0]

        color = sns.color_palette('deep')[0]

        bars = ax.barh(
            objs,
            values,
            color=color,
            alpha=0.5,
        )

        mean_value = values.mean()

        ax.axvline(mean_value, color='red', linestyle='dashed')

        ax.set_xlabel(f'Validation {metric.upper()} Values')
        ax.set_title(f'2 class classification LOOCV {metric.upper()} score')

        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2,
                    f'{width:.2f}',
                    va='center')

        ax.text(mean_value+0.15, ax.get_ylim()[1],
                f'Mean {metric.upper()}= {mean_value:.2f}',
                va='top',
                ha='center',
                color='red')

    plt.tight_layout()
    plt.close()

# Assuming valid_info and polymer_name_dict are defined elsewhere in the code
values = _get_resistance_binary_values(valid_info, polymer_name_dict, "smiles_polymer", is_rate=True,is_reverse=True)
visualize_all_metrics(dfs, values)

visualize_all_metrics(dfs, values)

# FIGURE 3: Feature Distribution Analysis

# *******************************
# Constants for batch adjustment of font size
# *******************************
BASE_FONT_SIZE = 40
TITLE_FONT_SIZE = int(BASE_FONT_SIZE * 1.25)    # e.g.: 50
LABEL_FONT_SIZE = int(BASE_FONT_SIZE * 1.125)     # e.g.: 45
TICK_FONT_SIZE = BASE_FONT_SIZE                  # e.g.: 40
ANNOTATION_FONT_SIZE = BASE_FONT_SIZE - 2         # e.g.: 38

plt.rcParams.update({
    'font.size': BASE_FONT_SIZE,
    'axes.titlesize': TITLE_FONT_SIZE,
    'axes.labelsize': LABEL_FONT_SIZE,
    'xtick.labelsize': TICK_FONT_SIZE,
    'ytick.labelsize': TICK_FONT_SIZE,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'pdf.fonttype': 42
})

def draw_polymer_graph(ax: Axes):
    df_opt = dfs["auc"]
    df_polymer = df_opt.iloc[0:1, 0:39]
    df_polymer = df_polymer.sort_values(by=df_polymer.index[0], axis=1, ascending=False)

    polymer_list = [
        re.sub(r'^cv_|_(micro_f1|macro_f1|auc|accuracy|fpr|fnr)$', '', col)
        for col in df_polymer.columns
    ]
    df_new = df_polymer.copy()
    df_new.columns = polymer_list

    # Example of converting to DataFrame if values is Series
    values_df = values.to_frame().T if isinstance(values, pd.Series) else values
    columns = values_df.columns
    new_columns = [re.search(r'\((.*?)\)', col).group(1) if '(' in col else col for col in columns]

    df_new.columns = [col.replace('cv_', '') for col in df_new.columns]
    df_new = df_new[new_columns]
    df_new.columns = columns

    objs = df_new.columns.tolist()
    values_series = df_new.iloc[0]
    color = sns.color_palette('deep')[0]
    
    bars = ax.barh(objs, values_series, color=color, alpha=0.5)
    mean_value = values_series.mean()
    ax.axvline(mean_value, color='red', linestyle='dashed')
    
    ax.set_xlabel('Leave one group out CV AUC values', fontsize=LABEL_FONT_SIZE)
    ax.set_title('2 class classification LOGOCV AUC score for polymer', fontsize=TITLE_FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    
    if valid_info is not None and polymer_name_dict is not None:
        ytick_labels = ax.get_yticklabels()
        for text_obj in ytick_labels:
            label = text_obj.get_text()
            key_found = None
            for key, info_dict in polymer_name_dict.items():
                if info_dict.get("jp_abbr", key) == label:
                    key_found = key
                    break
            if key_found is not None:
                try:
                    cb_val = valid_info.loc[valid_info["smiles_polymer"] == key_found, "crystalinity_binary"].iloc[0]
                    cb_int = int(cb_val)
                except Exception:
                    cb_int = 1
                text_obj.set_color("red" if cb_int == 0 else "black")
    
    for bar, val in zip(bars, values_series):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                va='center', fontsize=ANNOTATION_FONT_SIZE)
    
    ax.text(mean_value - 0.15, ax.get_ylim()[1]-0.1, f'Mean AUC= {mean_value:.2f}',
            va='top', ha='center', color='red', fontsize=TICK_FONT_SIZE)

def draw_solvent_graph(ax: Axes):
    df_opt_solvent = dfs["auc"]
    df_solvent = df_opt_solvent.loc[:, df_opt_solvent.columns.str.startswith('cv_solv_cluster')]
    df_solvent = df_solvent.sort_index(axis=1, 
                                       key=lambda x: x.str.extract(r'(\d+)').astype(int)[0],
                                       ascending=True)
    solv_cluster_list = [re.sub(r'^cv_solv_cluster', 'cluster', col) for col in df_solvent.columns]
    solv_cluster_list = [re.sub(r'_(micro_f1|macro_f1|auc|accuracy|fpr|fnr)$', '', col) for col in solv_cluster_list]
    df_solvent.columns = solv_cluster_list

    df_new_solvent = df_solvent[df_solvent.columns[::-1]]
    objs_solvent = df_new_solvent.columns.tolist()
    values_solvent = df_new_solvent.iloc[0]
    color = sns.color_palette('deep')[0]
    
    bars_solvent = ax.barh(objs_solvent, values_solvent, color=color, alpha=0.5)
    mean_value_solvent = values_solvent.mean()
    ax.axvline(mean_value_solvent, color='red', linestyle='dashed')
    
    ax.set_xlabel('Leave one group out CV AUC values', fontsize=LABEL_FONT_SIZE)
    ax.set_title('2 class classification LOGOCV AUC score for solvent', fontsize=TITLE_FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    
    for bar, val in zip(bars_solvent, values_solvent):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                va='center', fontsize=ANNOTATION_FONT_SIZE)
    
    ax.text(mean_value_solvent - 0.15, ax.get_ylim()[1]-0.1, f'Mean AUC = {mean_value_solvent:.2f}',
            va='top', ha='center', color='red', fontsize=TICK_FONT_SIZE)

fig3 = plotting.generate_figure3_combined(
    dfs=dfs,
    values=values,
    valid_info=valid_info,
    valid_info_cluster=valid_info_cluster,
    average_df=average_df,
    polymer_name_dict=polymer_name_dict,
    save_dir=REPORT_DIR_MAIN
)
plt.show()

all_features_combinations = [
    FF_solvent_wo_polars_cols + FF_polymer_cols, #0
    FF_solvent_cols + FF_polymer_cols, #1
    FF_solvent_wo_polars_cols + FF_polymer_cols + crystalinity_binary_col, #2
    FF_solvent_cols + FF_polymer_cols + crystalinity_binary_col, #3
    FF_solvent_wo_polars_cols + FF_polymer_cols + radonpy_polymer_cols, #4
    FF_solvent_cols + FF_polymer_cols + radonpy_polymer_cols, #5
    FF_solvent_wo_polars_cols + FF_polymer_cols + crystalinity_binary_col+ radonpy_polymer_cols, #6
    FF_solvent_cols + FF_polymer_cols + crystalinity_binary_col+ radonpy_polymer_cols, #7
    FF_solvent_wo_polars_cols + FF_polymer_cols + chi_col, #8
    FF_solvent_cols + FF_polymer_cols + chi_col, #9
    FF_solvent_wo_polars_cols + FF_polymer_cols + crystalinity_binary_col + chi_col, #10
    FF_solvent_cols + FF_polymer_cols + crystalinity_binary_col + chi_col, #11
    FF_solvent_wo_polars_cols + FF_polymer_cols + radonpy_polymer_cols + chi_col, #12
    FF_solvent_cols + FF_polymer_cols + radonpy_polymer_cols + chi_col, #13
    FF_solvent_wo_polars_cols + FF_polymer_cols + crystalinity_binary_col+ radonpy_polymer_cols+ chi_col, #14
    FF_solvent_cols + FF_polymer_cols + crystalinity_binary_col+ radonpy_polymer_cols+ chi_col, #15
]

features = list(all_features_combinations[feature_No]) 
print("all",len(features))

# Columns to exclude
exclude_cols = ['n_atom_radonpy_polymer', 'mol_weight_radonpy_polymer']
# 2. Use list comprehension
features = [col for col in features if col not in exclude_cols]
print("excluded",len(features))

train_data = valid_data_cluster.copy()[list(features)]
X_train = train_data[features]

# RESISTANCE MODEL: Training or Loading
dir_name = MODEL_DIR

if TRAIN_RESISTANCE_MODEL:
    # Train and save new model
    print("Training new resistance prediction model...")
    gbdt_model = train.train_and_save_resistance_model(X_train, y_train, params, dir_name)
    print(f"Model saved to {dir_name}")
else:
    # Load pre-trained model
    print("Loading pre-trained resistance prediction model...")
    model_filepath = os.path.join(dir_name, "chemical_resistance_model.pkl")
    gbdt_model = train.load_model(model_filepath)

# Model Analysis and Evaluation
features = list(all_features_combinations[feature_No])
exclude_cols = ['n_atom_radonpy_polymer', 'mol_weight_radonpy_polymer']
features = [col for col in features if col not in exclude_cols]
print(f"Features: {len(features)}")

train_data = valid_data_cluster.copy()[list(features)]
X_train = train_data[features]

result_dir = os.path.join(REPORT_DIR_SUB, "gbdt_model_result")
plotting.save_model_results(gbdt_model, X_train, y_train, save_dir=result_dir, threshold=0.9)

# Resin Evaluation
metrics_save_dir = os.path.join(REPORT_DIR_SUB, "gbdt_model_result")
metrics.save_metrics_by_group(
    model=gbdt_model, 
    X_data=X_train, 
    y_data=y_train, 
    group_info=train_info, 
    group_column="symbol", 
    save_dir=metrics_save_dir, 
    num_classes=num_classes, 
    threshold=0.5,
    new_columns=new_columns
)

metrics.save_metrics_by_group(
    model=gbdt_model, 
    X_data=X_train, 
    y_data=y_train, 
    group_info=train_info, 
    group_column="cluster_labels", 
    save_dir=metrics_save_dir, 
    num_classes=num_classes, 
    threshold=0.5,
    new_columns=new_columns
)

# Feature Importance
fi_save_dir = os.path.join(REPORT_DIR_SUB, "gbdt_model_result", "feature_importance")
plotting.plot_feature_importance(gbdt_model, X_train.columns.tolist(), save_dir=fi_save_dir, top_n=50)

# MIC calculation for feature importance analysis
if not SKIP_MIC:
    print(f"Calculating MIC for feature importance (experimental data)...")
    mic_dir = os.path.join(REPORT_DIR_SUB,"gbdt_model_result","feature_importance")
    os.makedirs(mic_dir, exist_ok=True)

    filename="mic_feature_importance_all.jpg"
    
    print(f"  Dataset: Experimental data (n={len(X_train)})")
    print(f"  Features: {len(X_train.columns)} features")

    mic.calculate_and_save_mic(gbdt_model, X_train, y_train, dir_name=mic_dir, top_n=100, filename=filename, n_jobs=N_JOBS)
else:
    print("Skipping MIC calculation for feature importance")

# Chi Parameter Surrogate Model
features = all_features_combinations[feature_No]
exclude_cols = ['n_atom_radonpy_polymer', 'mol_weight_radonpy_polymer']
features = [col for col in features if col not in exclude_cols]
data = valid_data_cluster.copy()[list(features)]
chi_explanatory_cols = FF_solvent_cols + FF_polymer_cols + radonpy_polymer_cols
chi_explanatory_cols = [col for col in chi_explanatory_cols if col not in exclude_cols]
X = data[chi_explanatory_cols]
y = data[chi_col]
model_name = "XGBoost"

print("=== 10-Fold Cross Validation ===")
save_dir = os.path.join(REPORT_DIR_SUB,f"chi_model_{model_name}")
os.makedirs(save_dir, exist_ok=True)

# Run chi parameter CV using validation module
chi_cv_results = validation.run_chi_parameter_cv(
    X=X,
    y=y,
    model=train.get_chi_model(model_name),
    n_splits=10,
    random_state=42
)

r2_scores = chi_cv_results['r2_scores']
mae_scores = chi_cv_results['mae_scores']
rmse_scores = chi_cv_results['rmse_scores']
cv_results = chi_cv_results['cv_results']
parity_data = chi_cv_results['parity_data']
mean_r2 = chi_cv_results['mean_r2']
std_r2 = chi_cv_results['std_r2']

mean_mae = chi_cv_results['mean_mae']
std_mae = chi_cv_results['std_mae']

mean_rmse = chi_cv_results['mean_rmse']
std_rmse = chi_cv_results['std_rmse']

mean_row = pd.DataFrame({
    'Fold': ['Mean'],
    'R2': [mean_r2],
    'MAE': [mean_mae],
    'RMSE': [mean_rmse]
})

std_row = pd.DataFrame({
    'Fold': ['Std'],
    'R2': [std_r2],
    'MAE': [std_mae],
    'RMSE': [std_rmse]
})

cv_results_df = pd.DataFrame(cv_results)
cv_results_df = pd.concat([cv_results_df, mean_row, std_row], ignore_index=True)
cv_results_df.to_csv(os.path.join(save_dir, 'cv_results.csv'), index=False)

final_model = train.get_chi_model(model_name)
final_model.fit(X, y)

y_pred_full = final_model.predict(X)
r2_full = r2_score(y, y_pred_full)
mae_full = mean_absolute_error(y, y_pred_full)
rmse_full = np.sqrt(mean_squared_error(y, y_pred_full))

print("\n=== Evaluation Results on All Data ===")
print(f"R2 (All Data): {r2_full:.4f}")
print(f"MAE (All Data): {mae_full:.4f}")
print(f"RMSE (All Data): {rmse_full:.4f}")

full_data_results = {
    'R2': [r2_full],
    'MAE': [mae_full],
    'RMSE': [rmse_full]
}
full_data_results_df = pd.DataFrame(full_data_results)
full_data_results_df.to_csv(os.path.join(save_dir, 'full_data_results.csv'), index=False)

model_path = os.path.join(MODEL_DIR, "chi_parameter_model.pkl")
dump(final_model, model_path)
print(f"\nModel saved: {model_path}")

# Accuracy Verification of Chi Parameter Surrogate Model
features = all_features_combinations[feature_No]
exclude_cols = ['n_atom_radonpy_polymer', 'mol_weight_radonpy_polymer']
features = [col for col in features if col not in exclude_cols]
data = valid_data_cluster.copy()[list(features)]
chi_explanatory_cols = FF_solvent_cols + FF_polymer_cols + radonpy_polymer_cols
chi_explanatory_cols = [col for col in chi_explanatory_cols if col not in exclude_cols]
X = data[chi_explanatory_cols]
y = data[chi_col]
model_name = "XGBoost"
kf = KFold(n_splits=10, shuffle=True, random_state=42)
r2_train_list, mae_train_list, rmse_train_list = [], [], []
r2_val_list, mae_val_list, rmse_val_list = [], [], []
cv_results = []
save_dir = os.path.join(REPORT_DIR_SUB, f"chi_model_{model_name}")
os.makedirs(save_dir, exist_ok=True)

print("=== 10-Fold Cross Validation Started ===")
fig, axes = plt.subplots(5, 2, figsize=(15, 25))
axes = axes.flatten()

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model = train.get_chi_model(model_name)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    y_val_pred = model.predict(X_val)
    r2_val = r2_score(y_val, y_val_pred)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2_train_list.append(r2_train)
    mae_train_list.append(mae_train)
    rmse_train_list.append(rmse_train)
    
    r2_val_list.append(r2_val)
    mae_val_list.append(mae_val)
    rmse_val_list.append(rmse_val)
    cv_results.append({'Fold': fold + 1, 'Train_R2': r2_train, 'Train_MAE': mae_train,
                       'Train_RMSE': rmse_train, 'Val_R2': r2_val, 'Val_MAE': mae_val, 'Val_RMSE': rmse_val})
    ax = axes[fold]
    ax.scatter(y_train, y_train_pred, color='blue', alpha=0.5, label='Train')
    ax.scatter(y_val, y_val_pred, color='red', alpha=0.5, label='Validation')
    min_val = min(np.min(y_train), np.min(y_val))
    max_val = max(np.max(y_train), np.max(y_val))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    ax.set_title(
        f"Fold {fold+1}\n"
        f"Train: R$^2$ = {r2_train:.4f}, MAE = {mae_train:.4f}, RMSE = {rmse_train:.4f}\n"
        f"Val: R$^2$ = {r2_val:.4f}, MAE = {mae_val:.4f}, RMSE = {rmse_val:.4f}"
    )
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend()

plt.tight_layout()
plt.close()

cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv(os.path.join(save_dir, 'cv_results.csv'), index=False)

summary_dict = {
    'Metric': ['R2', 'MAE', 'RMSE'],
    'Train_Mean': [np.mean(r2_train_list), np.mean(mae_train_list), np.mean(rmse_train_list)],
    'Train_Std': [np.std(r2_train_list), np.std(mae_train_list), np.std(rmse_train_list)],
    'Val_Mean': [np.mean(r2_val_list), np.mean(mae_val_list), np.mean(rmse_val_list)],
    'Val_Std': [np.std(r2_val_list), np.std(mae_val_list), np.std(rmse_val_list)]
}

summary_df = pd.DataFrame(summary_dict)
summary_csv_path = os.path.join(save_dir, 'cv_summary.csv')
summary_df.to_csv(summary_csv_path, index=False)
print(f"CV summary results saved: {summary_csv_path}")

# CHI MODEL: Training or Loading
features = all_features_combinations[feature_No]
exclude_cols = ['n_atom_radonpy_polymer', 'mol_weight_radonpy_polymer']
features = [col for col in features if col not in exclude_cols]
data = valid_data_cluster.copy()[list(features)]
chi_explanatory_cols = FF_solvent_cols + FF_polymer_cols + radonpy_polymer_cols
chi_explanatory_cols = [col for col in chi_explanatory_cols if col not in exclude_cols]
X = data[chi_explanatory_cols]
y = data[chi_col]
model_path = os.path.join(MODEL_DIR, "chi_parameter_model.pkl")

if TRAIN_CHI_MODEL:
    # === Train new chi parameter model ===
    print("Training new chi parameter prediction model...")
    model_name = "XGBoost"
    final_model = train.train_and_save_chi_model(
        X_train=X,
        y_train=y,
        save_dir=MODEL_DIR,
        model_name=model_name,
        model_filename="chi_parameter_model.pkl",
        random_state=42
    )
    
    # Evaluate final model
    y_pred_full = final_model.predict(X)
    r2_full = r2_score(y, y_pred_full)
    mae_full = mean_absolute_error(y, y_pred_full)
    rmse_full = np.sqrt(mean_squared_error(y, y_pred_full))
    
    print("\n=== Model Evaluation Results ===")
    print(f"R2: {r2_full:.4f}")
    print(f"MAE: {mae_full:.4f}")
    print(f"RMSE: {rmse_full:.4f}")
    
    # Save results to CSV
    full_data_results = {
        'R2': [r2_full],
        'MAE': [mae_full],
        'RMSE': [rmse_full]
    }
    full_data_results_df = pd.DataFrame(full_data_results)
    full_data_results_path = os.path.join(REPORT_DIR_SUB, f"chi_model_{model_name}", 'full_data_results.csv')
    full_data_results_df.to_csv(full_data_results_path, index=False)
    print(f"Evaluation results saved: {full_data_results_path}")
else:
    # === Load pre-trained chi model ===
    print("Loading pre-trained chi parameter prediction model...")
    final_model = train.load_model(model_path)
    print(f"Model loaded from: {model_path}")

# CRYSTALLINITY MODEL: Training or Loading Results
data = valid_data_cluster.copy()[list(features)]
data.drop_duplicates(subset=FF_polymer_cols, inplace=True)
CL_explanatory_cols = radonpy_polymer_cols
exclude_cols = ['n_atom_radonpy_polymer', 'mol_weight_radonpy_polymer']
CL_explanatory_cols = [col for col in CL_explanatory_cols if col not in exclude_cols]
X = data[CL_explanatory_cols]
y = data[crystalinity_binary_col]

if TRAIN_CRYSTALLINITY_MODEL:
    print(f"Training crystallinity model with 100 random seeds using {N_JOBS} parallel jobs...")
    import warnings
    from sklearn.exceptions import DataConversionWarning
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    n_iterations = 100
    cv_results_dict = validation.run_crystallinity_cv(
        X=X,
        y=y,
        n_iterations=n_iterations,
        n_splits=9,
        n_jobs=N_JOBS  # Parallel execution using --n-jobs argument
    )
    
    all_accuracies = cv_results_dict['accuracies']
    all_precisions = cv_results_dict['precisions']
    all_recalls = cv_results_dict['recalls']
    all_f1_scores = cv_results_dict['f1_scores']
    all_roc_aucs = cv_results_dict['roc_aucs']
    all_fold_roc_aucs = cv_results_dict['all_fold_roc_aucs']
    all_fold_f1_scores = cv_results_dict['all_fold_f1_scores']
    
    all_roc_aucs = [x for x in all_roc_aucs if not np.isnan(x)]
    print("\n=== Statistics for Evaluation Metrics over 100 Random Seeds ===")
    def print_stats(metric_name, values):
        print(f"{metric_name}:")
        print(f"  Mean: {np.mean(values):.4f}")
        print(f"  Std: {np.std(values):.4f}")
        print(f"  Max: {np.max(values):.4f}")
        print(f"  Min: {np.min(values):.4f}\n")

    print_stats("Accuracy", all_accuracies)
    print_stats("Precision", all_precisions)
    print_stats("Recall", all_recalls)
    print_stats("F1 Score", all_f1_scores)
    if all_roc_aucs:
        print_stats("ROC-AUC", all_roc_aucs)
    else:
        print("ROC-AUC: No valid folds for computation.")

    df_results = pd.DataFrame({
        "Random State": np.arange(len(all_accuracies)),
        "Accuracy": all_accuracies,
        "Precision": all_precisions,
        "Recall": all_recalls,
        "F1 Score": all_f1_scores,
        "ROC-AUC": all_roc_aucs + [np.nan] * (len(all_accuracies) - len(all_roc_aucs))  # Align length
    })

    df_summary = df_results.drop(columns=["Random State"]).describe().T
    df_summary = df_summary.rename(columns={"mean": "Mean", "std": "Std Dev", "50%": "Median", "min": "Min", "max": "Max"})
    df_summary = df_summary[["Mean", "Std Dev", "Median", "Min", "Max"]]
    save_dir = os.path.join(REPORT_DIR_SUB, "cl_model")
    os.makedirs(save_dir, exist_ok=True)

    csv_filename = os.path.join(save_dir, "cl_model_evaluation_results.csv")
    df_results.to_csv(csv_filename, index=False)
    print(f"✅ Evaluation results saved at {csv_filename}")
    summary_filename = os.path.join(save_dir, "cl_model_evaluation_summary.csv")
    df_summary.to_csv(summary_filename, index=True)
    print(f"✅ Summary statistics saved at {summary_filename}")
    fold_results_filename = os.path.join(save_dir, "cl_model_fold_results.pkl")
    import pickle
    with open(fold_results_filename, 'wb') as f:
        pickle.dump({'all_fold_roc_aucs': all_fold_roc_aucs, 'all_fold_f1_scores': all_fold_f1_scores}, f)
    print(f"✅ Fold-level results saved at {fold_results_filename}")
else:
    print("Loading pre-computed crystallinity model evaluation results...")
    save_dir = os.path.join(REPORT_DIR_SUB, "cl_model")
    csv_filename = os.path.join(save_dir, "cl_model_evaluation_results.csv")
    summary_filename = os.path.join(save_dir, "cl_model_evaluation_summary.csv")
    
    if os.path.exists(csv_filename) and os.path.exists(summary_filename):
        df_results = pd.read_csv(csv_filename)
        df_summary = pd.read_csv(summary_filename, index_col=0)
        print(f"Results loaded from {csv_filename}")
        print(f"Summary loaded from {summary_filename}")
        
        fold_results_filename = os.path.join(save_dir, "cl_model_fold_results.pkl")
        if os.path.exists(fold_results_filename):
            import pickle
            with open(fold_results_filename, 'rb') as f:
                fold_data = pickle.load(f)
            all_fold_roc_aucs = fold_data['all_fold_roc_aucs']
            all_fold_f1_scores = fold_data['all_fold_f1_scores']
            print(f"Fold-level results loaded from {fold_results_filename}")
        else:
            all_fold_roc_aucs = []
            all_fold_f1_scores = []
            print(f"WARNING: Fold-level results not found at {fold_results_filename}")
    else:
        print(f"Warning: Pre-computed results not found in {save_dir}")
        print("Please set TRAIN_CRYSTALLINITY_MODEL=True to generate results")

# SUPPLEMENTARY FIGURE S4: Cross-Validation Performance Scatter Plots
if all_fold_roc_aucs and all_fold_f1_scores:
    plotting.generate_supplementary_s4_cv_performance(
        all_fold_roc_aucs=all_fold_roc_aucs,
        all_fold_f1_scores=all_fold_f1_scores,
        save_dir=REPORT_DIR_SUB,
        font_size=16
    )
    plt.show()
else:
    print("Skipping S4 generation: No cross-validation fold data available.")
    print("To generate S4, run with --train-crystallinity flag.")

# Crystallinity Model Training on All Data
model = train.train_and_save_crystallinity_model(
    X_train=X,
    y_train=y,
    save_dir=MODEL_DIR,
    model_filename="crystallinity_model.pkl",
    random_state=42
)

feature_importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

print("\n=== Feature Importance ===")
print(importance_df)

save_dir = os.path.join(REPORT_DIR_SUB, "cl_model")
os.makedirs(save_dir, exist_ok=True)
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance (RandomForestClassifier)')
plt.gca().invert_yaxis()
for index, value in enumerate(importance_df['Importance']):
    plt.text(value, index, f"{value:.3f}", va='center')
feature_importance_path = os.path.join(save_dir, "feature_importance.png")
plt.tight_layout()
plt.savefig(feature_importance_path)
plt.close()

print(f"Feature importance plot saved: {feature_importance_path}")

y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]
cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()
fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr_value = fn / (fn + tp) if (fn + tp) > 0 else 0
f1 = f1_score(y, y_pred)
roc_auc = roc_auc_score(y, y_proba)

print("\n=== Evaluation Metrics ===")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"FPR (False Positive Rate): {fpr_value:.4f}")
print(f"FNR (False Negative Rate): {fnr_value:.4f}")

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.text(0.5, -0.1, f'F1: {f1:.3f}\nROC-AUC: {roc_auc:.3f}\nFPR: {fpr_value:.3f}\nFNR: {fnr_value:.3f}',
         transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
conf_matrix_path = os.path.join(save_dir, "confusion_matrix.png")
plt.savefig(conf_matrix_path)
plt.close()
print(f"Confusion Matrix saved: {conf_matrix_path}")

fpr, tpr, thresholds = roc_curve(y, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', color='blue')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.text(0.6, 0.2, f'F1: {f1:.3f}\nFPR: {fpr_value:.3f}\nFNR: {fnr_value:.3f}',
         transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
roc_curve_path = os.path.join(save_dir, "roc_curve.png")
plt.savefig(roc_curve_path)
plt.close()
print(f"ROC Curve saved: {roc_curve_path}")

# Load Acetone and Virtual Polymer Data
model_path = os.path.join(MODEL_DIR, "crystallinity_model.pkl")
model = train.load_model(model_path)
print("✅ Model loaded successfully!")
if _POLYMER_MPK_LOADED:
    df_acetone_vPolymer = _df_polymer_mpk.copy()
    print("✓ Using preprocessed polymer_mpk_dataset")
else:
    df_acetone_vPolymer = pd.read_csv("../data/vPolymer_vSolvent/MPK_vPolymer_40971_data.csv",index_col=0)

rename_cols = {col: col.replace("_resin", "_polymer") for col in df_acetone_vPolymer.columns if "_resin" in col}
df_acetone_vPolymer = df_acetone_vPolymer.rename(columns=rename_cols)
df_acetone_vPolymer["crystalinity_binary"] = model.predict(df_acetone_vPolymer[CL_explanatory_cols])
df_acetone_vPolymer["crystalinity_prob"] = model.predict_proba(df_acetone_vPolymer[CL_explanatory_cols])[:,1]
df_acetone_vPolymer.value_counts("crystalinity_binary")

# MIC for Figure 4: Crystallinity prediction model
if not SKIP_MIC:
    print("Calculating MIC for Figure 4 (Crystallinity prediction)...")
    dir_name = os.path.join(REPORT_DIR_SUB, "cl_model", "analysis_MIC")
    os.makedirs(dir_name, exist_ok=True)
    filename = "mic_analysis_cl.jpg"
    X_cl = df_acetone_vPolymer[CL_explanatory_cols].copy()
    y_cl = df_acetone_vPolymer["crystalinity_prob"].copy()
    print(f"  Dataset size: {len(X_cl)}, Features: {len(CL_explanatory_cols)}")
    mic.calculate_and_save_mic(
        model, X_cl, y_cl, 
        dir_name=dir_name, top_n=20, 
        filename=filename, n_jobs=N_JOBS
    )
else:
    print("Skipping MIC calculation for Figure 4")

# FIGURE 4: Feature Importance Analysis
mic_csv_path = os.path.join(REPORT_DIR_SUB, "cl_model", "analysis_MIC", "mic_analysis_cl_mic_scores.csv")
fig4 = plotting.generate_figure4_mic_and_rg(
    mic_csv_path=mic_csv_path,
    df_acetone_vPolymer=df_acetone_vPolymer,
    save_dir=REPORT_DIR_MAIN,
    base_fontsize=12
)
plt.show()

# Load Polyethylene and Virtual Solvent Data
model_path = os.path.join(MODEL_DIR, "chi_parameter_model.pkl")
final_model = load(model_path)
print("✅ Chi model loaded successfully!")
chi_explanatory_cols = FF_solvent_cols + FF_polymer_cols + radonpy_polymer_cols
exclude_cols = ['n_atom_radonpy_polymer', 'mol_weight_radonpy_polymer']
chi_explanatory_cols = [col for col in chi_explanatory_cols if col not in exclude_cols]
if _PE_SOLVENT_LOADED:
    df_PE_vSolv = _df_pe_solvent.copy()
    print("✓ Using preprocessed pe_solvent_dataset")
else:
    df_PE_vSolv = pd.read_csv("../data/vPolymer_vSolvent/PE_vsolv_9828_data.csv",index_col=0)

rename_cols = {col: col.replace("_resin", "_polymer") for col in df_PE_vSolv.columns if "_resin" in col}
df_PE_vSolv = df_PE_vSolv.rename(columns=rename_cols)
df_PE_vSolv["chi"] = final_model.predict(df_PE_vSolv[chi_explanatory_cols])
plt.hist(df_PE_vSolv["chi"], bins=30)

# Calculate Chi Parameter for All Experimental Resins and Virtual Solvents
data = valid_data_cluster[FF_polymer_cols + radonpy_polymer_cols]
info = valid_info_cluster.reindex(data.index)

common_cols = data.columns.intersection(info.columns)
info_reduced = info.drop(columns=common_cols)
df_merged = pd.merge(info_reduced, data, left_index=True, right_index=True)
df_merged = df_merged[["symbol"] + FF_polymer_cols + radonpy_polymer_cols + ["crystalinity_binary"]]
df_merged_unique = df_merged.drop_duplicates(subset="symbol")
df_dict = {}
for _, row in df_merged_unique.iterrows():
    symbol_key = row["symbol"]
    df_temp = df_PE_vSolv.copy()
    cols_to_replace = FF_polymer_cols + radonpy_polymer_cols + ["crystalinity_binary"]
    df_temp.loc[:, cols_to_replace] = row[cols_to_replace].values
    df_temp["chi"] = final_model.predict(df_temp[chi_explanatory_cols])
    df_temp["resistance_pred"] = gbdt_model.predict_proba(df_temp[features])[:, 1]
    df_temp["resistance_pred_binary"] = (df_temp["resistance_pred"] >= 0.9).astype(int)
    df_dict[symbol_key] = df_temp
def plot_chi_hist_by_binary(df, symbol, chi_col="chi", binary_col="resistance_pred_binary",
                             xlim=None, save_path=None, font_size=20):
    """Plot chi histogram by binary classification."""
    tick_fontsize = font_size
    chi = df[chi_col]
    binary = df[binary_col]
    if xlim is None:
        chi_min, chi_max = chi.min(), chi.max()
        margin = 0.1 * (chi_max - chi_min) if chi_max != chi_min else 1
        xlim = (chi_min - margin, chi_max + margin)

    bins = np.linspace(xlim[0], xlim[1], 51)
    chi_0 = chi[binary == 0]
    chi_1 = chi[binary == 1]
    n_0, n_1 = len(chi_0), len(chi_1)
    n_all = n_0 + n_1
    plt.figure(figsize=(10, 8))
    plt.hist(chi_0, bins=bins, alpha=0.3, label=f"resistant (n={n_0})", density=True, color="green")
    plt.hist(chi_1, bins=bins, alpha=0.3, label=f"non-resistant (n={n_1})", density=True, color="red")
    plt.title(f"{symbol}", fontsize=font_size)
    plt.xlabel("χ", fontsize=font_size)
    plt.ylabel("Normalized frequency", fontsize=font_size)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=font_size)
    plt.xlim(xlim)
    plt.tight_layout()
    plt.close()

all_chi_dir = os.path.join(REPORT_DIR_MAIN, "all_chi")
os.makedirs(all_chi_dir, exist_ok=True)

for symbol, df in df_dict.items():
    final_symbol = plotting.get_final_symbol(symbol, symbol_order)
    display_symbol = final_symbol.replace("solv", "solvent", 1) if final_symbol.startswith("solv") else final_symbol
    save_file = f"chi_hist_{final_symbol}.svg"
    save_path = os.path.join(all_chi_dir, save_file)
    plot_chi_hist_by_binary(
        df,
        symbol=symbol,
        chi_col="chi",
        binary_col="resistance_pred_binary",
        xlim=None,  # Auto-adjust based on each symbol's data
        save_path=save_path,
        font_size=18
    )

# SUPPLEMENTARY FIGURE S6: Chi Parameter Histogram by Polymer
fig_s6_group1, fig_s6_group2 = plotting.generate_supplementary_s6_chi_histograms(
    df_dict=df_dict,
    symbol_order=symbol_order,
    save_dir=REPORT_DIR_SUB,
    base_fontsize=10,
    chi_col="chi",
    binary_col="resistance_pred_binary"
)
plt.show()
plt.clf()

# FIGURE 9: Crystallinity Parameter Analysis
fig9 = plotting.generate_figure9_chi_analysis(
    valid_data_cluster=valid_data_cluster,
    valid_info_cluster=valid_info_cluster,
    gbdt_model=gbdt_model,
    all_features_combinations=all_features_combinations,
    feature_No=feature_No,
    df_dict=df_dict,
    symbol_order=symbol_order,
    save_dir=REPORT_DIR_MAIN,
    base_fontsize=20
)
plt.show()

df_PE_vSolv.loc[:, "crystalinity_binary"] = 1  # PE is crystalline
df_PE_vSolv["resistance_pred"] = gbdt_model.predict_proba(df_PE_vSolv[features])[:, 1]
df_PE_vSolv["resistance_pred_binary"] = (df_PE_vSolv["resistance_pred"] >= 0.9).astype(int)

df_PE_vSolv.shape

X_train = df_PE_vSolv[features]
y_train = df_PE_vSolv["resistance_pred"]

# MIC CALCULATION FOR FIGURE 7
if not SKIP_MIC:
    print("Calculating MIC for Figure 7 (Resistance on virtual solvent)...")
    dir_name = os.path.join(REPORT_DIR_SUB, "gbdt_model_result", "analysis_solvent")
    os.makedirs(dir_name, exist_ok=True)
    filename = "mic_analysis_polar.jpg"
    print(f"  Dataset size: {len(X_train)}, Features: {len(features)}")
    mic.calculate_and_save_mic(gbdt_model, X_train, y_train, dir_name=dir_name, top_n=20, filename=filename, n_jobs=N_JOBS)
else:
    print("Skipping MIC calculation for Figure 7")

# Load MIC results
dir_name = os.path.join(REPORT_DIR_SUB, "gbdt_model_result", "analysis_solvent")
filename = os.path.join(dir_name, "mic_analysis_polar_mic_scores.csv")
df_vsolvent_MIC = pd.read_csv(filename)
df_vsolvent_MIC.head()

# FIGURE 7c: Chi Parameter Distribution by Polymer Type
fig7c = plotting.generate_figure7c_chi_mic(
    df_vsolvent_MIC=df_vsolvent_MIC,
    FF_solvent_cols=FF_solvent_cols,
    save_dir=REPORT_DIR_MAIN,
    base_font_size=18
)
plt.show()

# FIGURE 7b: Chi Parameter Relationship Analysis
fig7b_plot, fig7b_heatmap = plotting.generate_figure7b_resistance_analysis(
    df_PE_vSolv=df_PE_vSolv,
    FF_solvent_cols=FF_solvent_cols,
    save_dir=REPORT_DIR_MAIN,
    base_font_size=28
)
plt.show()

# FIGURE 7a: Chi Parameter Prediction Model Performance (Heatmap)
df_heatmap = df_PE_vSolv.copy()
df_heatmap.sort_values("resistance_pred", inplace=True)
fig7a = plotting.generate_figure7a_heatmap(
    df_heatmap=df_heatmap,
    FF_solvent_cols=FF_solvent_cols,
    save_dir=REPORT_DIR_MAIN,
    base_font_size=28
)
plt.show()

# FIGURE 7d: Chi Parameter Feature Importance (PCA Scatter Plot)
fig7d = plotting.generate_figure7d_pca_scatter(
    df_PE_vSolv=df_PE_vSolv,
    features=FF_solvent_cols,
    save_dir=REPORT_DIR_MAIN,
    base_font_size=24
)
plt.show()

# SUPPLEMENTARY FIGURE S5: Force Field Parameter Average Plots
df_vPolymer = df_heatmap.copy()
fig_s5_4panel, fig_s5_6panel = plotting.generate_supplementary_s5_ff_parameter_analysis(
    df_vPolymer=df_vPolymer,
    FF_solvent_cols=FF_solvent_cols,
    save_dir=REPORT_DIR_SUB,
    base_font_size=14,
    Th_high=0.9,
    Th_low=0.1,
    alpha=0.2
)
plt.show()
plt.clf()

# FIGURE 8: Comprehensive Model Comparison
df_vPolymer = df_heatmap.copy()
fig8 = plotting.generate_figure8_model_comparison(
    df_vPolymer=df_vPolymer,
    FF_solvent_cols=FF_solvent_cols,
    save_dir=REPORT_DIR_MAIN,
    base_font_size=12,
    Th_high=0.9,
    Th_low=0.1
)
plt.show()
plt.clf()

# Analysis of Chi Parameter for Virtual Solvents
df_PE_vSolv["resistance_pred_binary"] = (df_PE_vSolv["resistance_pred"] >= 0.9).astype(int)
chi = df_PE_vSolv["chi"]
binary = df_PE_vSolv["resistance_pred_binary"]
chi_0 = chi[binary == 0]
chi_1 = chi[binary == 1]
n_0, n_1 = len(chi_0), len(chi_1)
n_all = n_0 + n_1
bins = np.linspace(-3, 5, 21)
plt.figure(figsize=(6, 4))
plt.hist(chi_0, bins=bins, alpha=0.5, label=f"Pred 0 (n={n_0})", density=True)
plt.hist(chi_1, bins=bins, alpha=0.5, label=f"Pred 1 (n={n_1})", density=True)

plt.title(f"chi histogram by resistance_pred_binary (Total n={n_all})")
plt.xlabel("chi")
plt.ylabel("Density")
plt.legend()
plt.xlim(-3, 5)
plt.tight_layout()
plt.close()

plot_chi_hist_by_binary(
    df_PE_vSolv,
    symbol="PE",  # Added required symbol argument
    chi_col="chi",
    binary_col="resistance_pred_binary",
    xlim=(-3, 7),
    save_path=os.path.join(REPORT_DIR_MAIN, "fig5_chi_hist.jpg"),
    font_size=20
)

def plot_chi_hist_from_features(df_features, df_info, model, all_features,
    exclude_cols: list = ['n_atom_radonpy_polymer', 'mol_weight_radonpy_polymer'],
    chi_col: str = "chi", binary_col: str = "resistance_binary",
    xlim: tuple = (-0.5, 7), save_path: str = None, font_size: int = 20):
    """Plot chi histogram from features DataFrame."""
    features = [col for col in all_features if col not in exclude_cols]
    train_data = df_features.copy()[features]
    X_train_exp = train_data[features]
    train_data["resistance_pred"] = model.predict_proba(X_train_exp)[:, 1]
    label = df_info[binary_col]
    CL_label = df_info.get("crystalinity_binary", None)
    chi = X_train_exp[chi_col]
    chi_0 = chi[label == 0]
    chi_1 = chi[label == 1]
    n_0, n_1 = len(chi_0), len(chi_1)
    n_all = n_0 + n_1
    bins = np.linspace(xlim[0], xlim[1], 51)
    plt.figure(figsize=(10, 8))
    plt.hist(chi_0, bins=bins, color="green", alpha=0.3, label=f"resistant (n={n_0})", density=True)
    plt.hist(chi_1, bins=bins, color="red", alpha=0.3, label=f"non-resistant (n={n_1})", density=True)
    tick_fontsize = font_size
    plt.title(f"{chi_col} histogram by {binary_col} (Total n={n_all})", fontsize=font_size)
    plt.xlabel("χ", fontsize=font_size)
    plt.ylabel("Density", fontsize=font_size)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=font_size)
    plt.xlim(xlim)
    plt.tight_layout()
    plt.close()

features_list = list(all_features_combinations[feature_No])
save_filepath = os.path.join(REPORT_DIR_MAIN, "fig5_chi_hist_exp.jpg")
plot_chi_hist_from_features(
    df_features = valid_data_cluster,
    df_info = valid_info_cluster,
    model = gbdt_model,
    all_features = features_list,
    exclude_cols = ['n_atom_radonpy_polymer', 'mol_weight_radonpy_polymer'],
    chi_col = "chi",
    binary_col = "resistance_binary",
    xlim = (-3, 7),
    save_path = save_filepath,
    font_size = 20
)

# Plot Chi Parameter for All Experimental Resins and Solvents
features_list = list(all_features_combinations[feature_No])
all_chi_COSMO_dir = os.path.join(REPORT_DIR_MAIN, "all_chi_COSMO")
os.makedirs(all_chi_COSMO_dir, exist_ok=True)
unique_symbols = valid_info_cluster["symbol"].unique()
for symbol in unique_symbols:
    idx = valid_info_cluster[valid_info_cluster["symbol"] == symbol].index
    df_features_subset = valid_data_cluster.loc[idx].copy()
    df_info_subset = valid_info_cluster.loc[idx].copy()
    save_filepath = os.path.join(all_chi_COSMO_dir, f"fig5_chi_hist_exp_{symbol}.svg")
    plot_chi_hist_from_features(
        df_features = df_features_subset,
        df_info = df_info_subset,
        model = gbdt_model,
        all_features = features_list,
        exclude_cols = ['n_atom_radonpy_polymer', 'mol_weight_radonpy_polymer'],
        chi_col = "chi",
        binary_col = "resistance_binary",
        xlim = (-3, 7),
        save_path = save_filepath,
        font_size = 20
    )

# Analysis of Experimental Data and Chi Parameter
features = list(all_features_combinations[feature_No])
exclude_cols = ['n_atom_radonpy_polymer', 'mol_weight_radonpy_polymer']
features = [col for col in features if col not in exclude_cols]
train_data = valid_data_cluster.copy()[list(features)]
X_train_exp = train_data[features]
train_data["resistance_pred"] = gbdt_model.predict_proba(X_train_exp)[:, 1]
label = valid_info_cluster["resistance_binary"]
CL_label = valid_info_cluster["crystalinity_binary"]
chi_0 = X_train_exp["chi"][label == 0]
chi_1 = X_train_exp["chi"][label == 1]
chi_0_CL0 = X_train_exp["chi"][(label == 0) & (CL_label == 0)]
chi_1_CL0 = X_train_exp["chi"][(label == 1) & (CL_label == 0)]
chi_0_CL1 = X_train_exp["chi"][(label == 0) & (CL_label == 1)]
chi_1_CL1 = X_train_exp["chi"][(label == 1) & (CL_label == 1)]
n_all_0, n_all_1 = len(chi_0), len(chi_1)
n_all = n_all_0 + n_all_1
n_CL0_0, n_CL0_1 = len(chi_0_CL0), len(chi_1_CL0)
n_CL0_all = n_CL0_0 + n_CL0_1
n_CL1_0, n_CL1_1 = len(chi_0_CL1), len(chi_1_CL1)
n_CL1_all = n_CL1_0 + n_CL1_1
bins = np.linspace(-3, 5, 21)
fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
axes[0].hist(chi_0, bins=bins, alpha=0.5, label=f"Label 0 (n={n_all_0})", density=True)
axes[0].hist(chi_1, bins=bins, alpha=0.5, label=f"Label 1 (n={n_all_1})", density=True)
axes[0].set_title(f"Overall {n_all} data")
axes[0].set_xlabel("chi")
axes[0].set_ylabel("Density")
axes[0].legend()
axes[1].hist(chi_0_CL0, bins=bins, alpha=0.5, label=f"Label 0 (n={n_CL0_0})", density=True)
axes[1].hist(chi_1_CL0, bins=bins, alpha=0.5, label=f"Label 1 (n={n_CL0_1})", density=True)
axes[1].set_title(f"Amorphous polymer {n_CL0_all} data")
axes[1].set_xlabel("chi")
axes[1].legend()
axes[2].hist(chi_0_CL1, bins=bins, alpha=0.5, label=f"Label 0 (n={n_CL1_0})", density=True)
axes[2].hist(chi_1_CL1, bins=bins, alpha=0.5, label=f"Label 1 (n={n_CL1_1})", density=True)
axes[2].set_title(f"Crystalline polymer {n_CL1_all} data")
axes[2].set_xlabel("chi")
axes[2].legend()
for ax in axes:
    ax.set_xlim(-3, 5)
plt.tight_layout()
plt.close()

# FIGURE 5: Resistance Prediction Results
if not SKIP_MIC:
    print("Calculating MIC for Figure 5 (Resistance prediction)...")
    mic_dir = os.path.join(REPORT_DIR_SUB, "gbdt_model_result", "analysis_polymer", "MIC_results")
    os.makedirs(mic_dir, exist_ok=True)
    features_list = list(all_features_combinations[feature_No])
    exclude_cols = ['n_atom_radonpy_polymer', 'mol_weight_radonpy_polymer']
    features_list = [col for col in features_list if col not in exclude_cols]
    if "chi" not in df_acetone_vPolymer.columns:
        df_acetone_vPolymer["chi"] = final_model.predict(df_acetone_vPolymer[chi_explanatory_cols])
    X_vpolymer = df_acetone_vPolymer[features_list].copy()
    y_resistance = gbdt_model.predict_proba(X_vpolymer)[:, 1]
    print(f"  Dataset size: {len(X_vpolymer)}, Features: {len(features_list)}")
    print("  - Calculating MIC for all data...")
    filename_all = "MPK_polymer_all_40971_data.jpg"
    mic.calculate_and_save_mic(
        gbdt_model, X_vpolymer, pd.Series(y_resistance), 
        dir_name=mic_dir, top_n=50, 
        filename=filename_all, n_jobs=N_JOBS
    )
    
    mask_crystal = df_acetone_vPolymer["crystalinity_binary"] == 1
    X_crystal = X_vpolymer[mask_crystal].copy()
    y_crystal = pd.Series(y_resistance[mask_crystal])
    print(f"  - Calculating MIC for crystalline data (n={len(X_crystal)})...")
    filename_crystal = "MPK_polymer_crystalinity_1_24361_data.jpg"
    mic.calculate_and_save_mic(
        gbdt_model, X_crystal, y_crystal,
        dir_name=mic_dir, top_n=50,
        filename=filename_crystal, n_jobs=N_JOBS
    )
    
    mask_amorph = df_acetone_vPolymer["crystalinity_binary"] == 0
    X_amorph = X_vpolymer[mask_amorph].copy()
    y_amorph = pd.Series(y_resistance[mask_amorph])
    print(f"  - Calculating MIC for amorphous data (n={len(X_amorph)})...")
    filename_amorph = "MPK_polymer_crystalinity_0_16610_data.jpg"
    mic.calculate_and_save_mic(
        gbdt_model, X_amorph, y_amorph,
        dir_name=mic_dir, top_n=50,
        filename=filename_amorph, n_jobs=N_JOBS
    )
    
    print("✅ MIC calculation for Figure 5 completed!")
else:
    print("Skipping MIC calculation for Figure 5")

if 'REPORT_DIR_MAIN' not in globals():
    REPORT_DIR_MAIN = "./"
features = list(all_features_combinations[feature_No])
exclude_cols = ['n_atom_radonpy_polymer', 'mol_weight_radonpy_polymer']
features = [col for col in features if col not in exclude_cols]
plt.rcParams.update({'font.size': 18})

filepath_all = os.path.join(REPORT_DIR_SUB, "gbdt_model_result", "analysis_polymer", "MIC_results", "MPK_polymer_all_40971_data_mic_scores.csv")
filepath_crystal = os.path.join(REPORT_DIR_SUB, "gbdt_model_result", "analysis_polymer", "MIC_results", "MPK_polymer_crystalinity_1_24361_data_mic_scores.csv")
filepath_noncrystal = os.path.join(REPORT_DIR_SUB, "gbdt_model_result", "analysis_polymer", "MIC_results", "MPK_polymer_crystalinity_0_16610_data_mic_scores.csv")

top_n = 10

def load_and_filter(filepath):
    df = pd.read_csv(filepath)
    df = df[~df['feature'].str.startswith('FF_')].copy()
    df['feature_clean'] = df['feature'].str.replace('_radonpy_polymer', '', regex=False)
    return df

df_all = load_and_filter(filepath_all)
df_crystal = load_and_filter(filepath_crystal)
df_noncrystal = load_and_filter(filepath_noncrystal)

top_features = df_all.sort_values(by='MIC', ascending=False).head(top_n)['feature']
def get_mic_values(df, top_features):
    df_sub = df[df['feature'].isin(top_features)].copy()
    return df_sub.set_index('feature')['MIC'].reindex(top_features).values
mic_all = get_mic_values(df_all, top_features)
mic_crystal = get_mic_values(df_crystal, top_features)
mic_noncrystal = get_mic_values(df_noncrystal, top_features)
feature_labels = df_all.set_index('feature').loc[top_features, 'feature_clean'].values
x = np.arange(top_n)
bar_width = 0.25

df_acetone_vPolymer["crystalinity_binary"] = model.predict(df_acetone_vPolymer[CL_explanatory_cols])
df_acetone_vPolymer["crystalinity_prob"] = model.predict_proba(df_acetone_vPolymer[CL_explanatory_cols])[:,1]
df_acetone_vPolymer["chi"] = final_model.predict(df_acetone_vPolymer[chi_explanatory_cols])
df_acetone_vPolymer["resistance_pred"] = gbdt_model.predict_proba(df_acetone_vPolymer[features])[:, 1]

plt.figure()
plt.hist(df_acetone_vPolymer["resistance_pred"], bins=100)
plt.tight_layout()
plt.close()

plt.rcParams.update({'font.size': 24})
features_list = list(all_features_combinations[feature_No])
exclude_cols = ['n_atom_radonpy_polymer', 'mol_weight_radonpy_polymer']
features_list = [col for col in features_list if col not in exclude_cols]
train_data = df_acetone_vPolymer.copy()[features_list]
X_train_exp = train_data[features_list]
train_data["resistance_pred"] = gbdt_model.predict_proba(X_train_exp)[:, 1]
CL_label = train_data["crystalinity_binary"]
pred_CL0 = train_data["resistance_pred"][CL_label == 0]
pred_CL1 = train_data["resistance_pred"][CL_label == 1]
n_CL0, n_CL1 = len(pred_CL0), len(pred_CL1)
n_all = n_CL0 + n_CL1
plt.figure(figsize=(8, 6))
plt.hist(pred_CL1, bins=20, alpha=0.8, color='lightblue', density=True,
         label=f'Crystalline (n={n_CL1})')
plt.hist(pred_CL0, bins=20, alpha=0.5, color='gray', density=True,
         label=f'Amorphos (n={n_CL0})')
plt.xlabel("Resistance probability", fontsize=24)
plt.ylabel("Relative frequency", fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.tight_layout()

filename = os.path.join(REPORT_DIR_MAIN, "fig5b_resistance_pred.svg")
plt.savefig(filename, dpi=300)
plt.close()

fig5 = plotting.generate_figure5_resistance_prediction(
    df_all_mic=df_all,
    df_crystal_mic=df_crystal,
    df_noncrystal_mic=df_noncrystal,
    pred_CL0=pred_CL0,
    pred_CL1=pred_CL1,
    n_all=n_all,
    save_dir=REPORT_DIR_MAIN
)
plt.show()

# Analysis of RadonPy properties
df_acetone_vPolymer["resistance_pred"]
df_acetone_vPolymer[features]
features = list(all_features_combinations[feature_No])
exclude_cols = ['n_atom_radonpy_polymer', 'mol_weight_radonpy_polymer']
features = [col for col in features if col not in exclude_cols]
train_data = df_acetone_vPolymer.copy()[list(features)]
X_train_exp = train_data[features]
CL_label = df_acetone_vPolymer["crystalinity_binary"]
param = "density_radonpy_polymer"
param_0 = X_train_exp[param][CL_label == 0]
param_1 = X_train_exp[param][CL_label == 1]
x_min, x_max = min(X_train_exp[param]), max(X_train_exp[param])
bins = np.linspace(x_min, x_max, 100)
fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
ax.hist(param_0, bins=bins, alpha=0.5, label="amor", density=True)
ax.hist(param_1, bins=bins, alpha=0.5, label="crys", density=True)
ax.set_title("Overall")
ax.set_ylabel("Density")
ax.set_xlabel(param)
ax.legend()
plt.xlim(x_min, x_max)
plt.tight_layout()
plt.close()

# FIGURE 6: Crystallinity Effect on Resistance
fig6 = plotting.generate_figure6_crystallinity_effect(
    df_acetone_vPolymer=df_acetone_vPolymer,
    x_col="density_radonpy_polymer",
    y_col="resistance_pred",
    save_dir=REPORT_DIR_MAIN,
    base_fontsize=20
)
plt.show()

print("Analysis complete!")

