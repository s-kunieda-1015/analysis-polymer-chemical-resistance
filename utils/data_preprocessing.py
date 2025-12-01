#!/usr/bin/env python3
"""
Data preprocessing module for chemical resistance analysis.

This module provides functions to create and load the three main datasets:
1. chemical_resistance_dataset - Experimental data for model training (~2231 rows)
2. polymer_MPK_dataset - Virtual polymer data with MPK solvent (40971 rows)
3. PE_solvent_dataset - PE polymer with virtual solvents (9828 rows)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import plotting
from .config import (
    ROOT_DIR,
    DATA_DIR,
    INTERIM_DIR,
    STD_THRESHOLD,
    get_feature_columns,
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Data directory for processed datasets (same as DATA_DIR in new structure)
PROCESSED_DATA_DIR = DATA_DIR
PROCESSED_DATA_DIR.mkdir(exist_ok=True, parents=True)

# RadonPy property columns to use
RADONPY_POLYMER_COLS: List[str] = [
    "density_radonpy_polymer",
    "Rg_radonpy_polymer",
    "Scaled_Rg_radonpy_polymer",
    "self-diffusion_radonpy_polymer",
    "Cp_radonpy_polymer",
    "Cv_radonpy_polymer",
    "compressibility_radonpy_polymer",
    "isentropic_compressibility_radonpy_polymer",
    "bulk_modulus_radonpy_polymer",
    "isentropic_bulk_modulus_radonpy_polymer",
    "volume_expansion_radonpy_polymer",
    "linear_expansion_radonpy_polymer",
    "r2_radonpy_polymer",
    "static_dielectric_const_radonpy_polymer",
    "dielectric_const_dc_radonpy_polymer",
    "nematic_order_parameter_radonpy_polymer",
    "refractive_index_radonpy_polymer",
]

# Polymers to exclude (only one class)
POLYMERS_TO_EXCLUDE: List[str] = ["PVAc", "CA"]

# Solvents to exclude (only 1 data point or all 0 or 1)
SOLVENTS_TO_EXCLUDE_ADDITIONAL: List[str] = [
    "サリチル酸メチル Methylsalicylate",
    "クロロブロモメタン",
    "メチルイソプロピルケトン",
    "臭化アセチル",
    "ピペリジン Piperidine",
    "オクタデカン",
    "アクリル酸メチル",
    "無水酢酸 Acetic anhydride",
    "酢酸 Acetic acid 100RT",
    "ギ酸 Formic acid 90RT",
    "ブタノール（二級） sec-Butanol",
]

# Polymer symbol to English name mapping
POLYMER_NAME_MAPPING: Dict[str, str] = {
    "PE": "Polyethylene",
    "PP": "Polypropylene",
    "PMP": "Polymethylpentene",
    "PIB": "Polyisobutylene",
    "PS": "Polystyrene",
    "PVA": "Polyvinyl alcohol",
    "PFA": "Perfluoroalkoxy alkane",
    "PMMA": "Polymethyl methacrylate",
    "PVC": "Polyvinyl chloride",
    "PVDF": "Polyvinylidene fluoride",
    "PVdC": "Polyvinylidene chloride",
    "PTFE": "Polytetrafluoroethylene",
    "ECTFE": "Ethylene chlorotrifluoroethylene",
    "ETFE": "Ethylene tetrafluoroethylene",
    "POM": "Polyoxymethylene",
    "CP": "Chlorinated polyether",
    "PSF": "Polysulfone",
    "PEEK": "Polyether ether ketone",
    "PPS": "Polyphenylene sulfide",
    "PET": "Polyethylene terephthalate",
    "PBT": "Polybutylene terephthalate",
    "PA6": "Nylon 6",
    "PA": "Polyamide",
    "PA12": "Nylon 12",
    "PC": "Polycarbonate",
    "PPCO": "Polypropylene copolymer",
    "FEP": "Fluorinated ethylene propylene",
}


# =============================================================================
# DATA CLASS FOR RETURN TYPES
# =============================================================================

@dataclass
class ChemicalResistanceDataset:
    """Container for chemical resistance dataset components."""
    valid_info_cluster: pd.DataFrame
    valid_data_cluster: pd.DataFrame
    feature_cols_dict: Dict[str, List[str]]
    polymer_name_dict: Dict[str, Dict[str, str]]
    solvent_name_dict: Dict[str, Dict[str, str]]
    radonpy_polymer_cols: List[str]


# =============================================================================
# CHEMICAL RESISTANCE DATASET FUNCTIONS
# =============================================================================

def _load_and_merge_base_data(script_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and merge the base data files with chi parameters.
    
    Parameters
    ----------
    script_dir : Path
        Directory where the main script is located
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        preprocessed_info (target variables and SMILES), preprocessed_data (features)
    """
    # 1. Load target variable and SMILES information
    preprocessed_info = pd.read_csv(
        PREPROCESSED_DIR / "merged_data_augmantation_4146.csv",
        index_col=0,
    )
    
    # Rename "resistance" column to "resistance_binary" and then invert (0->1, 1->0)
    preprocessed_info = preprocessed_info.rename(columns={"resistance": "resistance_binary"})
    preprocessed_info["resistance_binary"] = 1 - preprocessed_info["resistance_binary"]
    
    # Rename "name_resin" column to "name_polymer"
    if "name_resin" in preprocessed_info.columns:
        preprocessed_info = preprocessed_info.rename(columns={"name_resin": "name_polymer"})
    
    # 2. Load pre-calculated features
    preprocessed_data = pd.read_csv(
        PREPROCESSED_DIR / "merged_features_augmantation_4146.csv",
        index_col=0,
    )
    
    # 3. Join preprocess_info and preprocessed_data by index
    merged_info_data = preprocessed_info.join(preprocessed_data, how='inner')
    
    # 4. Load chi parameter data
    chi_path = ROOT_DIR / "data" / "chi_parameter" / "preprocessed_data_COSMO_B3-LYP_def2-SVP_20250207_unique.csv"
    df_chi = pd.read_csv(chi_path, index_col=0)
    
    # 5. Reset index to prepare for merging
    merged_info_data_reset = merged_info_data.reset_index()
    
    # 6. Inner join using SMILES information as key
    final_merged_df = pd.merge(
        merged_info_data_reset,
        df_chi,
        left_on=['smiles_solvent', 'smiles_resin'],
        right_on=['smiles_solvent_RAW', 'repeating_unit_RAW']
    )
    
    # 7. Restore the index before joining ("index")
    final_merged_df.set_index('index', inplace=True)
    
    # 8. Separate into preprocessed_info and preprocessed_data columns
    preprocess_info_columns = preprocessed_info.columns.tolist()
    preprocessed_data_columns = preprocessed_data.columns.tolist() + ["chi_TARGET"]
    
    preprocess_info = final_merged_df[preprocess_info_columns]
    preprocessed_data = final_merged_df[preprocessed_data_columns]
    
    # Rename "chi_TARGET" column to "chi"
    preprocessed_data.rename(columns={"chi_TARGET": "chi"}, inplace=True)
    
    # Rename "smiles_resin" to "smiles_polymer" in info
    if "smiles_resin" in preprocess_info.columns:
        preprocess_info = preprocess_info.rename(columns={"smiles_resin": "smiles_polymer"})
    
    # Rename feature columns: _resin → _polymer (for consistency)
    rename_cols = {col: col.replace("_resin", "_polymer") for col in preprocessed_data.columns if "_resin" in col}
    preprocessed_data = preprocessed_data.rename(columns=rename_cols)
    print(f"  ✓ Renamed {len(rename_cols)} feature columns from _resin to _polymer")
    
    print(f"✓ Data merged with chi parameters: {len(preprocess_info)} records, {preprocessed_data.shape[1]} features")
    
    return preprocess_info, preprocessed_data


def _filter_low_variance_features(data: pd.DataFrame, std_threshold: float = STD_THRESHOLD) -> pd.DataFrame:
    """
    Remove features with too small standard deviation.
    
    Parameters
    ----------
    data : pd.DataFrame
        Feature data
    std_threshold : float
        Threshold for standard deviation filtering
        
    Returns
    -------
    pd.DataFrame
        Filtered data with low-variance features removed
    """
    # Check properties of features
    is_unique = data.nunique() == 1
    n_single_value = sum(is_unique)
    n_valid = sum(~is_unique)
    print(f"  Features with single value: {n_single_value}")
    print(f"  Valid features: {n_valid}")
    
    # Remove features with too small standard deviation
    std = data.describe().loc["std"]
    valid_data = data.drop(columns=std.index[std < std_threshold])
    
    n_removed = data.shape[1] - valid_data.shape[1]
    print(f"✓ Removed {n_removed} low-variance features (std < {std_threshold})")
    
    return valid_data


def _exclude_polymers(valid_info: pd.DataFrame, valid_data: pd.DataFrame,
                      polymers_to_exclude: List[str] = POLYMERS_TO_EXCLUDE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Exclude specified polymers from the dataset.
    
    Parameters
    ----------
    valid_info : pd.DataFrame
        Info DataFrame
    valid_data : pd.DataFrame
        Feature DataFrame
    polymers_to_exclude : List[str]
        List of polymer symbols to exclude
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Filtered (valid_info, valid_data)
    """
    valid_info = valid_info[~valid_info["symbol"].isin(polymers_to_exclude)]
    valid_data = valid_data[valid_data.index.isin(valid_info.index)]
    
    valid_info.drop_duplicates(inplace=True)
    
    total_rows = len(valid_info)
    unique_solv = valid_info['solvent'].nunique()
    unique_polymer = valid_info['name_polymer'].nunique()
    print(f"✓ After polymer exclusion: {total_rows} records, {unique_solv} solvents, {unique_polymer} polymers")
    
    return valid_info, valid_data


def _exclude_solvents(valid_info: pd.DataFrame, valid_data: pd.DataFrame,
                      script_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Exclude specified solvents from the dataset.
    
    Parameters
    ----------
    valid_info : pd.DataFrame
        Info DataFrame
    valid_data : pd.DataFrame
        Feature DataFrame
    script_dir : Path
        Directory where the main script is located
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Filtered (valid_info, valid_data)
    """
    # Load solvent cleansing data
    solvent_cleansing_path = ROOT_DIR / "data" / "organic_solvent_cleansing" / "updated_data_with_state_v11.xlsx"
    df_solvent_cleansing = pd.read_excel(solvent_cleansing_path)
    
    # Get solvents with unknown state ("不明" means "unknown" in the data file)
    removes = df_solvent_cleansing[df_solvent_cleansing["state"] == "不明"]["solvent"].unique().tolist()
    
    # Add additional solvents to exclude
    removes += SOLVENTS_TO_EXCLUDE_ADDITIONAL
    
    valid_info = valid_info[~valid_info["solvent"].isin(removes)]
    valid_data = valid_data[valid_data.index.isin(valid_info.index)]
    
    valid_info.drop_duplicates(inplace=True)
    
    total_rows = len(valid_info)
    unique_solv = valid_info['solvent'].nunique()
    unique_polymer = valid_info['name_polymer'].nunique()
    print(f"✓ After solvent exclusion: {total_rows} records, {unique_solv} solvents, {unique_polymer} polymers")
    
    return valid_info, valid_data


def _add_crystallinity_labels(valid_info: pd.DataFrame, valid_data: pd.DataFrame,
                               script_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add crystallinity labels to polymers.
    
    Parameters
    ----------
    valid_info : pd.DataFrame
        Info DataFrame
    valid_data : pd.DataFrame
        Feature DataFrame
    script_dir : Path
        Directory where the main script is located
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Updated (valid_info, valid_data) with crystallinity labels
    """
    # Load crystallinity data
    crystalinity_path = script_dir / "20231117_39polymers_crystalinity.csv"
    df_crystalinity = pd.read_csv(crystalinity_path, index_col=0, encoding="Shift_JIS")
    # Rename column from Japanese "結晶性ラベル" (crystallinity label) to English
    df_crystalinity = df_crystalinity.rename(columns={"結晶性ラベル": "crystalinity_binary"})
    
    # Store original index
    original_index = valid_info.index
    
    # Merge operation
    valid_info_crystalinity = pd.merge(
        valid_info, 
        df_crystalinity[["symbol", "crystalinity_binary"]], 
        left_on="symbol", 
        right_on="symbol", 
        how="left"
    )
    
    # Set original index
    valid_info_crystalinity.index = original_index
    
    valid_data_crystalinity = valid_data.copy()
    valid_data_crystalinity["crystalinity_binary"] = valid_info_crystalinity["crystalinity_binary"]
    
    valid_info_crystalinity = valid_info_crystalinity.dropna(subset=["crystalinity_binary"])
    valid_data_crystalinity = valid_data_crystalinity.dropna(subset=["crystalinity_binary"])
    
    print(f"After adding crystallinity labels: {len(valid_info_crystalinity)} records")
    
    return valid_info_crystalinity, valid_data_crystalinity


def _add_radonpy_properties(valid_info: pd.DataFrame, valid_data: pd.DataFrame,
                            script_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add RadonPy MD-calculated properties.
    
    Parameters
    ----------
    valid_info : pd.DataFrame
        Info DataFrame
    valid_data : pd.DataFrame
        Feature DataFrame
    script_dir : Path
        Directory where the main script is located
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Updated (valid_info, valid_data) with RadonPy properties
    """
    # Load RadonPy data
    radonpy_path = script_dir / "Radonpy_ttest" / "output" / "merged_resin_data_r1.csv"
    df_radonpy = pd.read_csv(radonpy_path, index_col=0, encoding="Shift_JIS")
    
    # Add '_radonpy_polymer' to column names other than 'resin_name'
    df_radonpy.rename(columns=lambda x: x if x == 'resin_name' else f"{x}_radonpy_polymer", inplace=True)
    
    # Store original index
    original_index = valid_info.index
    
    # Merge operation
    valid_info_radonpy = pd.merge(
        valid_info, 
        df_radonpy, 
        left_on="symbol", 
        right_on="resin_name", 
        how="left"
    )
    
    # Set original index
    valid_info_radonpy.index = original_index
    
    valid_data_radonpy = valid_data.copy()
    # Add columns from RadonPy data
    for column in df_radonpy.columns:
        if column != 'resin_name':
            valid_data_radonpy[column] = valid_info_radonpy[column]
    
    print(f"After adding RadonPy properties: {len(valid_info_radonpy)} records")
    
    return valid_info_radonpy, valid_data_radonpy


def _load_name_dictionaries(script_dir: Path) -> Tuple[Dict, Dict, pd.DataFrame]:
    """
    Load polymer and solvent name dictionaries and rename data.
    
    Parameters
    ----------
    script_dir : Path
        Directory where the main script is located
        
    Returns
    -------
    Tuple[Dict, Dict, pd.DataFrame]
        polymer_name_dict, solvent_name_dict, df_rename_solvent
    """
    # Load dictionaries (use original filename resin_name_dict.json)
    with (PREPROCESSED_DIR / "resin_name_dict.json").open("r", encoding="utf-8") as f:
        polymer_name_dict = json.load(f)
    with (PREPROCESSED_DIR / "solvent_name_dict.json").open("r", encoding="utf-8") as f:
        solvent_name_dict = json.load(f)
    
    # Update polymer name dict with jp_abbr
    for value in polymer_name_dict.values():
        value['jp_abbr'] = f"{value['abbr']}"
    
    # Load solvent rename data
    df_rename_solvent = pd.read_excel(script_dir / "rename_solvent" / "solvent_rename_after.xlsx", index_col=0)
    
    # Create mapping between solvent and english columns
    solvent_to_english = dict(zip(df_rename_solvent['solvent'], df_rename_solvent['english']))
    
    # Update solvent_name_dict to use English names
    for smi, names_dict in solvent_name_dict.items():
        names_dict['jp_abbr'] = names_dict.pop('jp')
    
    for smi, names_dict in solvent_name_dict.items():
        jp_abbr = names_dict['jp_abbr']
        if jp_abbr in solvent_to_english:
            names_dict['jp_abbr'] = solvent_to_english[jp_abbr]
    
    return polymer_name_dict, solvent_name_dict, df_rename_solvent


def _create_solvent_clusters(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Create solvent clusters based on resistance ratio.
    
    Parameters
    ----------
    df_merged : pd.DataFrame
        Merged data with info and features
        
    Returns
    -------
    pd.DataFrame
        Solvent cluster data
    """
    # Calculate resistance_binary rate for each solvent
    percentage_df = (df_merged
                     .groupby('solvent')['resistance_binary']
                     .apply(plotting.calculate_percentage)
                     .unstack()
                     .reset_index()
                    )
    percentage_df.fillna(0, inplace=True)
    
    # Merge rates into original DataFrame
    merged_df = df_merged.merge(percentage_df, on='solvent', how='left')
    
    # Rename columns
    columns_rename = {'A': 'A_original', 0: 'resistance_0_(%)', 1: 'resistance_1_(%)'}
    merged_df.rename(columns=columns_rename, inplace=True)
    
    # Extract unique solvent data
    df_solv = merged_df.drop_duplicates(subset="solvent").reset_index(drop=True)
    df_solv_cluster = df_solv.copy()
    
    # Clustering process
    df_solv_cluster.sort_values("resistance_0_(%)", ascending=True, inplace=True)
    df_solv_cluster.reset_index(drop=True, inplace=True)
    
    # Calculate rank
    df_solv_cluster['rank'] = df_solv_cluster['resistance_0_(%)'].rank(method='min')
    
    # Split into 10 clusters
    clusters = np.array_split(df_solv_cluster, 10)
    
    # Assign labels to each cluster
    for i, cluster in enumerate(clusters):
        df_solv_cluster.loc[cluster.index, 'cluster_labels'] = i + 1
    
    # Convert cluster_labels to integer type
    df_solv_cluster['cluster_labels'] = df_solv_cluster['cluster_labels'].astype(int)
    
    print("Cluster distribution:")
    print(df_solv_cluster['cluster_labels'].value_counts().sort_index())
    
    return df_solv_cluster


def _finalize_cluster_data(valid_info: pd.DataFrame, valid_data: pd.DataFrame,
                           df_solv_cluster: pd.DataFrame,
                           df_rename_solvent: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Finalize the cluster data by merging and converting to English names.
    
    Parameters
    ----------
    valid_info : pd.DataFrame
        Info DataFrame
    valid_data : pd.DataFrame
        Feature DataFrame
    df_solv_cluster : pd.DataFrame
        Solvent cluster data
    df_rename_solvent : pd.DataFrame
        Solvent rename mapping
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        valid_info_cluster, valid_data_cluster
    """
    # Save index as temporary column
    valid_info['temp_index'] = valid_info.index
    
    # Merge with cluster data
    valid_info_cluster = valid_info.merge(
        df_solv_cluster[["smiles_solvent", "cluster_labels"]], 
        on="smiles_solvent", 
        how="inner"
    )
    
    # Create mapping for solvent -> english conversion
    solvent_to_english = dict(zip(df_rename_solvent['solvent'], df_rename_solvent['english']))
    
    # Replace solvent column with English names
    valid_info_cluster['solvent'] = valid_info_cluster['solvent'].replace(solvent_to_english)
    
    # Replace name_polymer column with English names (based on symbol)
    valid_info_cluster['name_polymer'] = valid_info_cluster['symbol'].map(POLYMER_NAME_MAPPING)
    print(f"✓ Converted name_polymer to English names")
    
    # Restore original index
    valid_info_cluster.set_index('temp_index', inplace=True, drop=True)
    valid_info_cluster = valid_info_cluster.sort_index()
    
    # Check for duplicate rows
    print("Duplicate rows in valid_info_cluster after merging:")
    print(valid_info_cluster[valid_info_cluster.index.duplicated(keep=False)])
    
    # Filter valid_data
    valid_data_cluster = valid_data.loc[valid_info_cluster.index]
    
    total_rows = len(valid_info_cluster)
    unique_solv = valid_info_cluster['solvent'].nunique()
    unique_polymer = valid_info_cluster['name_polymer'].nunique()
    print(f"✓ Final dataset: {total_rows} records, {unique_solv} solvents, {unique_polymer} polymers")
    
    return valid_info_cluster, valid_data_cluster


def create_chemical_resistance_dataset(script_dir: Optional[Path] = None) -> ChemicalResistanceDataset:
    """
    Create the chemical resistance dataset from raw data files.
    
    This function performs the following steps:
    1. Load and merge base data with chi parameters
    2. Filter low-variance features
    3. Exclude specified polymers and solvents
    4. Add crystallinity labels
    5. Add RadonPy properties
    6. Create solvent clusters
    7. Finalize data with English names
    
    Parameters
    ----------
    script_dir : Path, optional
        Directory where the main script is located.
        If None, uses notebooks_crystallinity_paper_3 directory.
        
    Returns
    -------
    ChemicalResistanceDataset
        Container with all dataset components
    """
    if script_dir is None:
        script_dir = ROOT_DIR / "notebooks_crystallinity_paper_3"
    
    print("=" * 60)
    print("Creating Chemical Resistance Dataset")
    print("=" * 60)
    
    # Step 1: Load and merge base data
    preprocessed_info, preprocessed_data = _load_and_merge_base_data(script_dir)
    
    # Step 2: Filter low-variance features
    valid_data = _filter_low_variance_features(preprocessed_data)
    
    # Step 3: Extract feature columns
    feature_cols_dict = get_feature_columns(valid_data)
    
    # Reindex valid_info to match valid_data
    valid_info = preprocessed_info.reindex(valid_data.index)
    
    print(f"Dataset summary:")
    print(f"  Resins: {len(valid_info['symbol'].unique())}")
    print(f"  Solvents: {len(valid_info['solvent'].unique())}")
    print(f"  Records: {len(valid_info)}")
    
    # Step 4: Exclude polymers
    valid_info, valid_data = _exclude_polymers(valid_info, valid_data)
    
    # Step 5: Exclude solvents
    valid_info, valid_data = _exclude_solvents(valid_info, valid_data, script_dir)
    
    # Step 6: Add crystallinity labels
    valid_info, valid_data = _add_crystallinity_labels(valid_info, valid_data, script_dir)
    
    # Step 7: Add RadonPy properties
    valid_info, valid_data = _add_radonpy_properties(valid_info, valid_data, script_dir)
    
    # Step 8: Load name dictionaries
    polymer_name_dict, solvent_name_dict, df_rename_solvent = _load_name_dictionaries(script_dir)
    
    # Step 9: Create merged data for clustering
    df_info = valid_info.copy()
    df_feats = valid_data.copy()
    
    # Remove duplicate rows
    df_info = df_info[~df_info.index.duplicated(keep='first')]
    df_feats = df_feats[~df_feats.index.duplicated(keep='first')]
    
    # Inner join
    df_merged = pd.merge(df_info, df_feats, how="inner", left_index=True, right_index=True)
    
    # Step 10: Create solvent clusters
    df_solv_cluster = _create_solvent_clusters(df_merged)
    
    # Step 11: Finalize cluster data
    valid_info_cluster, valid_data_cluster = _finalize_cluster_data(
        valid_info, valid_data, df_solv_cluster, df_rename_solvent
    )
    
    print("=" * 60)
    print("Chemical Resistance Dataset created successfully!")
    print("=" * 60)
    
    return ChemicalResistanceDataset(
        valid_info_cluster=valid_info_cluster,
        valid_data_cluster=valid_data_cluster,
        feature_cols_dict=feature_cols_dict,
        polymer_name_dict=polymer_name_dict,
        solvent_name_dict=solvent_name_dict,
        radonpy_polymer_cols=RADONPY_POLYMER_COLS,
    )


def save_chemical_resistance_dataset(
    dataset: ChemicalResistanceDataset,
    save_dir: Optional[Path] = None
) -> None:
    """
    Save the chemical resistance dataset to files.
    
    Saves:
    - chemical_resistance_dataset.csv/xlsx: Unified dataset (info + data)
    - chemical_resistance_metadata.json: All metadata in one file
    
    Parameters
    ----------
    dataset : ChemicalResistanceDataset
        Dataset to save
    save_dir : Path, optional
        Directory to save files. Defaults to data/processed/
    """
    if save_dir is None:
        save_dir = PROCESSED_DATA_DIR
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # ==========================================================================
    # Save unified dataset (info + data merged into one file)
    # ==========================================================================
    valid_info = dataset.valid_info_cluster.copy()
    valid_data = dataset.valid_data_cluster.copy()
    
    # Remove duplicate indices if any
    valid_info = valid_info[~valid_info.index.duplicated(keep='first')]
    valid_data = valid_data[~valid_data.index.duplicated(keep='first')]
    
    # Find common columns between info and data (e.g., crystalinity_binary, radonpy columns)
    common_cols = valid_info.columns.intersection(valid_data.columns).tolist()
    if common_cols:
        print(f"  Note: Found {len(common_cols)} common columns between info and data")
        # Remove common columns from data (keep them in info) to avoid _x/_y suffixes
        valid_data_for_merge = valid_data.drop(columns=common_cols)
    else:
        valid_data_for_merge = valid_data
    
    # Merge info and data (no duplicate columns now)
    df_unified = pd.merge(valid_info, valid_data_for_merge, left_index=True, right_index=True, how='inner')
    
    # Reorder columns: put important columns first
    priority_cols = [
        "symbol",
        "name_polymer",
        "repeating_unit",
        "smiles_polymer",
        "solvent",
        "smiles_solvent",
        "resistance_binary",
    ]
    # Get priority columns that exist in df_unified
    existing_priority_cols = [c for c in priority_cols if c in df_unified.columns]
    # Get remaining columns (preserve original order)
    remaining_cols = [c for c in df_unified.columns if c not in priority_cols]
    # Reorder columns
    df_unified = df_unified[existing_priority_cols + remaining_cols]
    
    # Save unified CSV and Excel
    unified_csv_path = save_dir / "chemical_resistance_dataset.csv"
    unified_xlsx_path = save_dir / "chemical_resistance_dataset.xlsx"
    
    df_unified.to_csv(unified_csv_path)
    df_unified.to_excel(unified_xlsx_path)
    
    print(f"✓ Saved chemical_resistance_dataset.csv: {len(df_unified)} rows, {len(df_unified.columns)} columns")
    print(f"✓ Saved chemical_resistance_dataset.xlsx")
    
    # ==========================================================================
    # Save all metadata in one unified JSON file
    # ==========================================================================
    metadata = {
        "info_columns": dataset.valid_info_cluster.columns.tolist(),
        "shared_columns": common_cols,
        "feature_columns": dataset.feature_cols_dict,
        "radonpy_polymer_cols": dataset.radonpy_polymer_cols,
        "polymer_name_dict": dataset.polymer_name_dict,
        "solvent_name_dict": dataset.solvent_name_dict,
    }
    
    metadata_path = save_dir / "chemical_resistance_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved chemical_resistance_metadata.json (all metadata unified)")
    print(f"  - info_columns: {len(metadata['info_columns'])} columns")
    print(f"  - shared_columns: {len(metadata['shared_columns'])} columns")
    print(f"  - feature_columns: {len(metadata['feature_columns'])} groups")
    print(f"  - radonpy_polymer_cols: {len(metadata['radonpy_polymer_cols'])} columns")
    print(f"  - polymer_name_dict: {len(metadata['polymer_name_dict'])} entries")
    print(f"  - solvent_name_dict: {len(metadata['solvent_name_dict'])} entries")
    
    print(f"\n✓ All files saved to: {save_dir}")


def load_chemical_resistance_dataset(load_dir: Optional[Path] = None) -> ChemicalResistanceDataset:
    """
    Load the chemical resistance dataset from files.
    
    This function loads:
    - chemical_resistance_dataset.csv: Unified dataset
    - chemical_resistance_metadata.json: All metadata in one file
    
    For backward compatibility, it also supports loading from separate JSON files
    if the unified metadata file doesn't exist.
    
    Parameters
    ----------
    load_dir : Path, optional
        Directory to load files from. Defaults to data/processed/
        
    Returns
    -------
    ChemicalResistanceDataset
        Loaded dataset
    """
    if load_dir is None:
        load_dir = PROCESSED_DATA_DIR
    
    load_dir = Path(load_dir)
    
    # Check for unified metadata file (new format)
    metadata_path = load_dir / "chemical_resistance_metadata.json"
    unified_path = load_dir / "chemical_resistance_dataset.csv"
    
    # Try to load from unified metadata first
    if metadata_path.exists():
        print("Loading from unified metadata (chemical_resistance_metadata.json)...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        info_cols = metadata["info_columns"]
        shared_cols = metadata["shared_columns"]
        feature_cols_dict = metadata["feature_columns"]
        radonpy_polymer_cols = metadata["radonpy_polymer_cols"]
        polymer_name_dict = metadata["polymer_name_dict"]
        solvent_name_dict = metadata["solvent_name_dict"]
        
        print(f"✓ Loaded chemical_resistance_metadata.json")
    else:
        # Fallback to separate JSON files (backward compatibility)
        print("Loading from separate metadata files (backward compatibility)...")
        
        info_cols_path = load_dir / "info_columns.json"
        shared_cols_path = load_dir / "shared_columns.json"
        feature_cols_path = load_dir / "feature_columns.json"
        radonpy_cols_path = load_dir / "radonpy_resin_cols.json"
        resin_dict_path = load_dir / "resin_name_dict.json"
        solvent_dict_path = load_dir / "solvent_name_dict.json"
        
        with open(info_cols_path, 'r', encoding='utf-8') as f:
            info_cols = json.load(f)
        
        if shared_cols_path.exists():
            with open(shared_cols_path, 'r', encoding='utf-8') as f:
                shared_cols = json.load(f)
        else:
            shared_cols = ['crystalinity_binary']
        
        with open(feature_cols_path, 'r', encoding='utf-8') as f:
            feature_cols_dict = json.load(f)
        
        with open(radonpy_cols_path, 'r', encoding='utf-8') as f:
            radonpy_polymer_cols = json.load(f)
        
        with open(resin_dict_path, 'r', encoding='utf-8') as f:
            polymer_name_dict = json.load(f)
        with open(solvent_dict_path, 'r', encoding='utf-8') as f:
            solvent_name_dict = json.load(f)
        
        print(f"✓ Loaded separate metadata files")
    
    # Load unified CSV
    if unified_path.exists():
        print("Loading unified dataset (chemical_resistance_dataset.csv)...")
        df_unified = pd.read_csv(unified_path, index_col=0)
        
        # Split into info and data
        existing_info_cols = [c for c in info_cols if c in df_unified.columns]
        info_only_cols = [c for c in existing_info_cols if c not in shared_cols]
        data_cols = [c for c in df_unified.columns if c not in info_only_cols]
        
        valid_info_cluster = df_unified[existing_info_cols].copy()
        valid_data_cluster = df_unified[data_cols].copy()
        
        print(f"✓ Loaded chemical_resistance_dataset.csv: {len(df_unified)} rows")
        print(f"✓ Split into info ({len(existing_info_cols)} cols) and data ({len(data_cols)} cols)")
    else:
        # Fallback to legacy format (separate info.csv + data.csv)
        print("Loading from legacy format (separate info/data files)...")
        info_path = load_dir / "chemical_resistance_info.csv"
        data_path = load_dir / "chemical_resistance_data.csv"
        
        valid_info_cluster = pd.read_csv(info_path, index_col=0)
        valid_data_cluster = pd.read_csv(data_path, index_col=0)
        
        print(f"✓ Loaded chemical_resistance_info.csv: {len(valid_info_cluster)} rows")
        print(f"✓ Loaded chemical_resistance_data.csv: {len(valid_data_cluster)} rows")
    
    print(f"✓ Loaded all files from: {load_dir}")
    
    return ChemicalResistanceDataset(
        valid_info_cluster=valid_info_cluster,
        valid_data_cluster=valid_data_cluster,
        feature_cols_dict=feature_cols_dict,
        polymer_name_dict=polymer_name_dict,
        solvent_name_dict=solvent_name_dict,
        radonpy_polymer_cols=radonpy_polymer_cols,
    )


# =============================================================================
# POLYMER MPK DATASET FUNCTIONS (Phase 2)
# =============================================================================

def load_polymer_mpk_raw_data() -> pd.DataFrame:
    """
    Load the raw polymer MPK dataset (40971 rows).
    
    Returns
    -------
    pd.DataFrame
        Raw polymer MPK data
    """
    data_path = ROOT_DIR / "data" / "vPolymer_vSolvent" / "MPK_vPolymer_40971_data.csv"
    df = pd.read_csv(data_path, index_col=0)
    print(f"✓ Loaded polymer_MPK_dataset: {len(df)} rows")
    return df


def save_polymer_mpk_dataset(df: pd.DataFrame, save_dir: Optional[Path] = None) -> None:
    """
    Save the polymer MPK dataset with info columns (chemical_resistance_dataset format).
    
    Info columns added:
    - symbol: "vP_{index}" (virtual polymer identifier)
    - repeating_unit: from vpolymer.csv (smiles_list column)
    - smiles_polymer: from vpolymer.csv (smiles_list column)
    - solvent: "Methyl propyl ketone" (fixed)
    - smiles_solvent: "CCCC(C)=O" (fixed, MPK SMILES)
    
    Note: Excel output is skipped (CSV only for large datasets).
    
    Parameters
    ----------
    df : pd.DataFrame
        Polymer MPK data
    save_dir : Path, optional
        Directory to save file
    """
    if save_dir is None:
        save_dir = PROCESSED_DATA_DIR
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Add info columns (chemical_resistance_dataset format)
    df_with_info = df.copy()
    
    # Rename columns: _resin → _polymer (for consistency with feature names)
    rename_cols = {col: col.replace("_resin", "_polymer") for col in df_with_info.columns if "_resin" in col}
    df_with_info = df_with_info.rename(columns=rename_cols)
    if rename_cols:
        print(f"  ✓ Renamed {len(rename_cols)} feature columns from _resin to _polymer")
    
    n_rows = len(df_with_info)
    
    # Generate virtual polymer identifiers based on index
    indices = df_with_info.index.tolist()
    
    # Load SMILES information from vpolymer.csv (source file for virtual polymers)
    # vpolymer.csv has smiles_list column with the polymer SMILES
    vpolymer_path = ROOT_DIR / "data" / "vPolymer_vSolvent" / "vpolymer.csv"
    repeating_units: List[str]
    if vpolymer_path.exists():
        try:
            # Only load the smiles_list column to save memory
            df_vpolymer = pd.read_csv(vpolymer_path, usecols=["smiles_list"])
            if len(df_vpolymer) == n_rows:
                repeating_units = df_vpolymer["smiles_list"].astype(str).tolist()
                print(f"  ✓ Loaded SMILES from vpolymer.csv: {len(repeating_units)} entries")
            else:
                print(f"  ⚠ vpolymer.csv row count mismatch: {len(df_vpolymer)} vs {n_rows}, using empty strings")
                repeating_units = [""] * n_rows
        except Exception as e:
            print(f"  ⚠ Failed to load vpolymer.csv: {e}, using empty strings")
            repeating_units = [""] * n_rows
    else:
        print(f"  ⚠ vpolymer.csv not found at {vpolymer_path}, using empty strings")
        repeating_units = [""] * n_rows
    
    info_data = {
        "symbol": [f"vP_{idx}" for idx in indices],  # Virtual polymer identifier
        "repeating_unit": repeating_units,  # Virtual polymer repeating unit SMILES
        "smiles_polymer": repeating_units,  # Use the same SMILES for polymer
        "solvent": ["Methyl propyl ketone"] * n_rows,  # Fixed solvent
        "smiles_solvent": ["CCCC(C)=O"] * n_rows,  # Fixed MPK SMILES
    }
    
    # Create info DataFrame
    info_df = pd.DataFrame(info_data, index=df_with_info.index)
    
    # Concatenate info columns at the front
    df_unified = pd.concat([info_df, df_with_info], axis=1)
    
    # Save CSV only (skip Excel for large datasets)
    csv_path = save_dir / "polymer_mpk_dataset.csv"
    df_unified.to_csv(csv_path)
    print(f"✓ Saved polymer_mpk_dataset.csv: {len(df_unified)} rows, {len(df_unified.columns)} columns")


def load_polymer_mpk_dataset(load_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the processed polymer MPK dataset.
    
    Parameters
    ----------
    load_dir : Path, optional
        Directory to load from
        
    Returns
    -------
    pd.DataFrame
        Polymer MPK data
    """
    if load_dir is None:
        load_dir = PROCESSED_DATA_DIR
    
    load_path = Path(load_dir) / "polymer_mpk_dataset.csv"
    df = pd.read_csv(load_path, index_col=0)
    print(f"✓ Loaded polymer_mpk_dataset.csv: {len(df)} rows")
    return df


# =============================================================================
# PE SOLVENT DATASET FUNCTIONS (Phase 2)
# =============================================================================

def load_pe_solvent_raw_data() -> pd.DataFrame:
    """
    Load the raw PE solvent dataset (9828 rows).
    
    Returns
    -------
    pd.DataFrame
        Raw PE solvent data
    """
    data_path = ROOT_DIR / "data" / "vPolymer_vSolvent" / "PE_vsolv_9828_data.csv"
    df = pd.read_csv(data_path, index_col=0)
    print(f"✓ Loaded PE_solvent_dataset: {len(df)} rows")
    return df


def save_pe_solvent_dataset(df: pd.DataFrame, save_dir: Optional[Path] = None) -> None:
    """
    Save the PE solvent dataset with info columns (chemical_resistance_dataset format).
    
    Info columns added:
    - symbol: "PE" (fixed)
    - repeating_unit: "CC" (fixed, PE repeating unit SMILES)
    - smiles_polymer: "CC" (fixed, PE SMILES)
    - smiles_solvent: from vsolv_9828_data_index.csv (smiles_solvent column)
    
    Note: Excel output is skipped (CSV only).
    
    Parameters
    ----------
    df : pd.DataFrame
        PE solvent data
    save_dir : Path, optional
        Directory to save file
    """
    if save_dir is None:
        save_dir = PROCESSED_DATA_DIR
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Add info columns (chemical_resistance_dataset format)
    df_with_info = df.copy()
    
    # Rename columns: _resin → _polymer (for consistency with feature names)
    rename_cols = {col: col.replace("_resin", "_polymer") for col in df_with_info.columns if "_resin" in col}
    df_with_info = df_with_info.rename(columns=rename_cols)
    if rename_cols:
        print(f"  ✓ Renamed {len(rename_cols)} feature columns from _resin to _polymer")
    
    n_rows = len(df_with_info)
    
    # Load SMILES information from vsolv_9828_data_index.csv (source file for virtual solvents)
    # vsolv_9828_data_index.csv has smiles_solvent column with the solvent SMILES
    vsolv_path = ROOT_DIR / "data" / "vPolymer_vSolvent" / "vsolv_9828_data_index.csv"
    solvent_smiles: List[str]
    if vsolv_path.exists():
        try:
            # Only load the smiles_solvent column to save memory
            df_vsolv = pd.read_csv(vsolv_path, usecols=["smiles_solvent"])
            if len(df_vsolv) == n_rows:
                solvent_smiles = df_vsolv["smiles_solvent"].astype(str).tolist()
                print(f"  ✓ Loaded SMILES from vsolv_9828_data_index.csv: {len(solvent_smiles)} entries")
            else:
                print(f"  ⚠ vsolv_9828_data_index.csv row count mismatch: {len(df_vsolv)} vs {n_rows}, using empty strings")
                solvent_smiles = [""] * n_rows
        except Exception as e:
            print(f"  ⚠ Failed to load vsolv_9828_data_index.csv: {e}, using empty strings")
            solvent_smiles = [""] * n_rows
    else:
        print(f"  ⚠ vsolv_9828_data_index.csv not found at {vsolv_path}, using empty strings")
        solvent_smiles = [""] * n_rows
    
    info_data = {
        "symbol": ["PE"] * n_rows,  # Fixed polymer
        "repeating_unit": ["CC"] * n_rows,  # Fixed PE repeating unit SMILES
        "smiles_polymer": ["CC"] * n_rows,  # Fixed PE SMILES
        "smiles_solvent": solvent_smiles,  # Virtual solvent SMILES
    }
    
    # Create info DataFrame
    info_df = pd.DataFrame(info_data, index=df_with_info.index)
    
    # Concatenate info columns at the front
    df_unified = pd.concat([info_df, df_with_info], axis=1)
    
    # Save CSV only (skip Excel)
    csv_path = save_dir / "pe_solvent_dataset.csv"
    df_unified.to_csv(csv_path)
    print(f"✓ Saved pe_solvent_dataset.csv: {len(df_unified)} rows, {len(df_unified.columns)} columns")


def load_pe_solvent_dataset(load_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the processed PE solvent dataset.
    
    Parameters
    ----------
    load_dir : Path, optional
        Directory to load from
        
    Returns
    -------
    pd.DataFrame
        PE solvent data
    """
    if load_dir is None:
        load_dir = PROCESSED_DATA_DIR
    
    load_path = Path(load_dir) / "pe_solvent_dataset.csv"
    df = pd.read_csv(load_path, index_col=0)
    print(f"✓ Loaded pe_solvent_dataset.csv: {len(df)} rows")
    return df


# =============================================================================
# ALL DATASETS LOADER (Phase 3)
# =============================================================================

def load_all_datasets(load_dir: Optional[Path] = None) -> Tuple[ChemicalResistanceDataset, pd.DataFrame, pd.DataFrame]:
    """
    Load all three datasets from processed files.
    
    Parameters
    ----------
    load_dir : Path, optional
        Directory to load files from. Defaults to data/processed/
        
    Returns
    -------
    Tuple[ChemicalResistanceDataset, pd.DataFrame, pd.DataFrame]
        chemical_resistance_dataset, polymer_mpk_df, pe_solvent_df
    """
    if load_dir is None:
        load_dir = PROCESSED_DATA_DIR
    
    print("=" * 60)
    print("Loading All Datasets")
    print("=" * 60)
    
    chemical_resistance = load_chemical_resistance_dataset(load_dir)
    polymer_mpk_df = load_polymer_mpk_dataset(load_dir)
    pe_solvent_df = load_pe_solvent_dataset(load_dir)
    
    print("=" * 60)
    print("All datasets loaded successfully!")
    print("=" * 60)
    
    return chemical_resistance, polymer_mpk_df, pe_solvent_df


def save_all_datasets(
    chemical_resistance: ChemicalResistanceDataset,
    polymer_mpk_df: pd.DataFrame,
    pe_solvent_df: pd.DataFrame,
    save_dir: Optional[Path] = None
) -> None:
    """
    Save all three datasets to processed files.
    
    Parameters
    ----------
    chemical_resistance : ChemicalResistanceDataset
        Chemical resistance dataset
    polymer_mpk_df : pd.DataFrame
        Polymer MPK data
    pe_solvent_df : pd.DataFrame
        PE solvent data
    save_dir : Path, optional
        Directory to save files
    """
    if save_dir is None:
        save_dir = PROCESSED_DATA_DIR
    
    print("=" * 60)
    print("Saving All Datasets")
    print("=" * 60)
    
    save_chemical_resistance_dataset(chemical_resistance, save_dir)
    save_polymer_mpk_dataset(polymer_mpk_df, save_dir)
    save_pe_solvent_dataset(pe_solvent_df, save_dir)
    
    print("=" * 60)
    print("All datasets saved successfully!")
    print("=" * 60)

