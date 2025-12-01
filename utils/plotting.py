#!/usr/bin/env python3
"""
Plotting functions for chemical resistance analysis paper figures.

This module contains all visualization and plotting functions used to generate
the main and supplementary figures for the paper. Functions are organized by
their purpose and the figures they support.

All functions import necessary configuration from utils.config.
"""

# Standard library imports
import os
import sys
from pathlib import Path
from typing import Any, Sequence, Optional, List, Tuple, Dict

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.ticker as mticker
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.transforms import blended_transform_factory
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns

# RDKit for chemical structure visualization
from rdkit import Chem
from rdkit.Chem import Draw

# Machine learning metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Statistical analysis
from minepy import MINE

# Progress bar
from tqdm import tqdm

# Import configuration
from .config import (
    REPORT_DIR_MAIN, REPORT_DIR_SUB, MODEL_DIR,
    BASE_FONT_SIZE, TITLE_FONT_SIZE, LABEL_FONT_SIZE, TICK_FONT_SIZE, ANNOTATION_FONT_SIZE,
    COLOR_RESISTANT, COLOR_NON_RESISTANT, ALPHA_HIST, ALPHA_FILL,
    Th_high, Th_low
)

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

def set_paper_style():
    """Set matplotlib style for paper-quality figures."""
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams.update({
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "axes.grid": False,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "font.size": 16,
        "font.family": "Arial",
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white"
    })

# =============================================================================
# BASIC PLOTTING UTILITIES
# =============================================================================

def save_as_png(fig: Figure, path: str) -> None:
    """
    Save a Figure object as a PNG file.
    
    Parameters
    ----------
    fig : Figure
        Matplotlib Figure object to save
    path : str
        Output file path
    """
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved to {path}")


def calculate_percentage(group):
    """
    Calculate percentage distribution for each group.
    
    Parameters
    ----------
    group : pd.Series or pd.DataFrame
        Data group to calculate percentages for
    
    Returns
    -------
    pd.Series
        Percentage values
    """
    total = len(group)
    return group.value_counts() / total * 100

# =============================================================================
# CHEMICAL STRUCTURE VISUALIZATION
# =============================================================================

def get_rdkit_image(smiles: str, size: tuple[int, int]=(100, 100)) -> np.ndarray:
    """
    Generate a chemical structure image from SMILES string using RDKit.
    
    Parameters
    ----------
    smiles : str
        SMILES string representing the molecule
    size : tuple[int, int], optional
        Image size (width, height), by default (100, 100)
    
    Returns
    -------
    np.ndarray
        Image as numpy array (RGB)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Return empty image if parsing fails
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        return img
    pil_img = Draw.MolToImage(mol, size=size)
    return np.array(pil_img)


def add_structure_images(
    ax: Axes,
    keys: list[str],
    smiles_name_dict: dict[str, dict[str, str]],
    image_func,
    image_zoom: float = 0.25,
    label_x: float = -0.1,
    image_x_offset: float = 0.1,
    image_y_offset: float = 0.0,
    special_labels: list = None
):
    """
    Add chemical structure images and labels to the left of an Axes.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib Axes to add images to
    keys : list[str]
        List of sample keys (SMILES strings)
    smiles_name_dict : dict
        Dictionary mapping keys to names and abbreviations
    image_func : callable
        Function to generate images from keys
    image_zoom : float, optional
        Zoom factor for images, by default 0.25
    label_x : float, optional
        X position for labels in axes coordinates, by default -0.1
    image_x_offset : float, optional
        X offset for images, by default 0.1
    image_y_offset : float, optional
        Y offset for images, by default 0.0
    special_labels : list, optional
        List of labels to be colored red (non-crystalline polymers)
    """
    # Hide y-axis tick labels
    ax.set_yticklabels([])
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    
    yticks = ax.get_yticks()
    
    # yticks are obtained AFTER ax.invert_yaxis() is called in _plot_resistance_binary_dists
    # So yticks order already matches the visual order (top to bottom)
    # We can directly zip yticks with keys
    for tick, key in zip(yticks, keys):
        label = smiles_name_dict.get(key, {}).get("jp_abbr", key)
        
        # Set color based on special_labels
        color = 'black'  # default
        if special_labels is not None and label in special_labels:
            color = 'red'
        
        ax.text(label_x, tick, label,
                transform=trans,
                ha='right', va='center', fontsize=10, color=color)
        img_arr = image_func(key)
        im = OffsetImage(img_arr, zoom=image_zoom)
        ab = AnnotationBbox(im,
                            (label_x + image_x_offset, tick + image_y_offset),
                            xycoords=trans,
                            frameon=False,
                            clip_on=False)
        ax.add_artist(ab)

# =============================================================================
# RESISTANCE DISTRIBUTION PLOTS
# =============================================================================

def _plot_resistance_binary_dists(
    info: pd.DataFrame,
    smiles_name_dict: dict[str, dict[str, str]],
    key_col: str,
    *,
    is_rate: bool = False,
    is_reverse: bool = False,
    is_rotate: bool = False,
    show_data_labels: bool = False,
    symbol_order: list = None
) -> tuple[Figure, Axes, list[str], list[str]]:
    """
    Plot binary resistance distribution as a bar graph.
    
    Parameters
    ----------
    info : pd.DataFrame
        DataFrame containing resistance data
    smiles_name_dict : dict
        Dictionary mapping keys to names and abbreviations
    key_col : str
        Column name for sample keys
    is_rate : bool, optional
        If True, show as rates (0-1), otherwise as counts, by default False
    is_reverse : bool, optional
        If True, reverse sort order, by default False
    is_rotate : bool, optional
        If True, rotate to vertical bars, by default False
    show_data_labels : bool, optional
        If True, display data values inside bars, by default False
    symbol_order : list, optional
        Predefined order of symbols (e.g., from df_sorted['symbol'])
        If None, will sort by resistance_binary ratio
    
    Returns
    -------
    tuple[Figure, Axes, list[str], list[str]]
        Figure, Axes, list of tick labels, and list of keys
    """
    # Obtain unique keys and their corresponding abbreviations
    key_samples = info.drop_duplicates(subset=key_col)
    keys: list[str] = key_samples[key_col].tolist()
    ticklabels = [smiles_name_dict[key]["jp_abbr"] for key in keys]

    # For each key, count the occurrences of resistance_binary values (0, 1)
    counts = pd.DataFrame({
        ylabel: info.loc[info[key_col] == key, "resistance_binary"].value_counts()
        for key, ylabel in zip(keys, ticklabels, strict=True)
    }).fillna(0).astype(int)

    # Convert to percentages
    rate_values = counts.div(counts.sum(axis=0), axis=1)
    
    # Sort columns based on symbol_order or resistance ratio
    if symbol_order is not None:
        # Use predefined symbol order (same as Figure 3)
        # Create mapping from symbol to ticklabel
        if 'symbol' in info.columns:
            # Get symbol for each key
            key_to_symbol = {}
            for key in keys:
                symbol_series = info[info[key_col] == key]['symbol']
                if len(symbol_series) > 0:
                    key_to_symbol[key] = symbol_series.iloc[0]
            
            # Create symbol to ticklabel mapping
            symbol_to_ticklabel = {}
            for key, ticklabel in zip(keys, ticklabels):
                if key in key_to_symbol:
                    symbol_to_ticklabel[key_to_symbol[key]] = ticklabel
            
            # Sort according to symbol_order
            sorted_columns = []
            for symbol in symbol_order:
                if symbol in symbol_to_ticklabel and symbol_to_ticklabel[symbol] in counts.columns:
                    sorted_columns.append(symbol_to_ticklabel[symbol])
        else:
            # Fallback: assume symbol_order contains ticklabels
            sorted_columns = [s for s in symbol_order if s in ticklabels]
    else:
        # Sort based on the proportion of non-resistant (1) values
        sorted_columns = rate_values.sort_values(by=1, ascending=not is_reverse, axis=1).columns.tolist()
    
    values = counts if not is_rate else rate_values
    values = values[sorted_columns]
    ticklabels = values.columns.tolist()
    objs: list[int] = values.index.tolist()
    
    # Reorder keys to match sorted ticklabels
    ticklabel_to_key = {smiles_name_dict[key]["jp_abbr"]: key for key in keys}
    keys = [ticklabel_to_key[ticklabel] for ticklabel in ticklabels if ticklabel in ticklabel_to_key]

    # Color mapping: 0 → green (resistant), 1 → red (non-resistant)
    color_mapping = {0: COLOR_RESISTANT, 1: COLOR_NON_RESISTANT}

    # Set the figure size based on the orientation (rotation)
    if is_rotate:
        fig, ax = plt.subplots(figsize=(len(ticklabels) * 0.4 + 1.5, 6))
    else:
        fig, ax = plt.subplots(figsize=(9.6, len(ticklabels) * 0.25 + 0.8))
        
    accumulations = pd.Series(0, index=ticklabels)

    # Draw a stacked bar graph for each resistance value (0, 1)
    for obj in objs:
        weights = values.loc[obj]
        color = color_mapping.get(obj, 'gray')
        current_accum = accumulations.copy()
        
        if is_rotate:
            rects = ax.bar(
                values.columns,
                weights,
                bottom=current_accum,
                label=obj,
                color=color,
                alpha=ALPHA_HIST,
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
            
            # Add data labels inside bars if requested
            if show_data_labels:
                for rect, w in zip(rects, weights):
                    if w > 0:
                        x_center = rect.get_x() + rect.get_width() / 2
                        y_center = rect.get_y() + rect.get_height() / 2
                        label_text = f"{w:.2f}" if is_rate else f"{int(w)}"
                        ax.text(x_center, y_center, label_text,
                                ha='center', va='center', color='black', fontsize=8)
        else:
            rects = ax.barh(
                values.columns,
                weights,
                left=current_accum,
                label=obj,
                color=color,
                alpha=ALPHA_HIST,
            )
            
            # Add data labels inside bars if requested
            if show_data_labels:
                for rect, w in zip(rects, weights):
                    if w > 0:
                        x_center = rect.get_x() + rect.get_width() / 2
                        y_center = rect.get_y() + rect.get_height() / 2
                        label_text = f"{w:.2f}" if is_rate else f"{int(w)}"
                        ax.text(x_center, y_center, label_text,
                                ha='center', va='center', color='black', fontsize=8)
        
        accumulations += weights

    # If displaying percentages, fix the axis limit to 1
    if is_rate:
        if is_rotate:
            ax.set_ylim(0, 1)
        else:
            ax.set_xlim(0, 1)
    
    # Set axis labels
    if is_rotate:
        label_str = "Resistance ratio" if is_rate else "Data count"
        ax.set_ylabel(label_str)
    else:
        label_str = "Resistance ratio" if is_rate else "Data count"
        ax.set_xlabel(label_str)
    
    # Invert y-axis for horizontal bars (non-rotated)
    if not is_rotate:
        ax.invert_yaxis()

    return fig, ax, ticklabels, keys


def _plot_resistance_binary_dists_fixed(
    info: pd.DataFrame,
    smiles_name_dict: dict[str, dict[str, str]],
    key_col: str,
    ax: Axes,
    *,
    sort_key: Sequence[Any] = [0],
    is_rate: bool = False,
    is_reverse: bool = False,
    is_rotate: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Internal function to plot resistance distribution on a fixed-size Axes.
    
    Parameters
    ----------
    info : pd.DataFrame
        DataFrame containing resistance data
    smiles_name_dict : dict
        Dictionary mapping keys to names and abbreviations
    key_col : str
        Column name for sample keys
    ax : Axes
        Matplotlib Axes to plot on
    sort_key : Sequence[Any], optional
        Keys to sort by, by default [0]
    is_rate : bool, optional
        If True, show as rates, by default False
    is_reverse : bool, optional
        If True, reverse sort order, by default False
    is_rotate : bool, optional
        If True, rotate to vertical bars, by default False
    
    Returns
    -------
    tuple[list[str], list[str]]
        Tick labels and sample keys
    """
    # ① Obtain unique sample keys and generate abbreviations for plotting labels
    key_samples = info.drop_duplicates(subset=key_col)
    keys: list[str] = key_samples[key_col].tolist()
    ticklabels = [smiles_name_dict.get(key, {}).get("jp_abbr", key) for key in keys]

    # ② Create a DataFrame of resistance_binary counts for each sample
    counts = pd.DataFrame({
        label: info.loc[info[key_col] == key, "resistance_binary"].value_counts()
        for key, label in zip(keys, ticklabels, strict=True)
    }).fillna(0).astype(int)
    counts = counts.reindex([0, 1])
    
    # ③ Calculate percentages and sort based on the proportion of 0
    rate_values = counts.div(counts.sum(axis=0), axis=1)
    sorted_columns = rate_values.sort_values(by=[0], ascending=False, axis=1).columns.tolist()
    values = counts if not is_rate else rate_values
    values = values[sorted_columns]
    ticklabels = values.columns.tolist()
    objs: list[int] = values.index.tolist()

    # ④ Create a stacked bar chart
    colors = [COLOR_RESISTANT, COLOR_NON_RESISTANT]
    accumulations = pd.Series(0, index=ticklabels)
    for i, obj in enumerate(objs):
        weights = values.loc[obj]
        current_accum = accumulations.copy()
        if is_rotate:
            rects = ax.bar(
                values.columns,
                weights,
                bottom=current_accum,
                label=obj,
                color=colors[i % len(colors)],
                alpha=ALPHA_HIST,
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
            for rect, w in zip(rects, weights):
                if w > 0:
                    x_center = rect.get_x() + rect.get_width() / 2
                    y_center = rect.get_y() + rect.get_height() / 2
                    label_text = f"{w:.2f}" if is_rate else f"{int(w)}"
                    ax.text(x_center, y_center, label_text,
                            ha='center', va='center', color='black', fontsize=8)
        else:
            rects = ax.barh(
                values.columns,
                weights,
                left=current_accum,
                label=obj,
                color=colors[i % len(colors)],
                alpha=ALPHA_HIST,
            )
            for rect, w in zip(rects, weights):
                if w > 0:
                    x_center = rect.get_x() + rect.get_width() / 2
                    y_center = rect.get_y() + rect.get_height() / 2
                    label_text = f"{w:.2f}" if is_rate else f"{int(w)}"
                    ax.text(x_center, y_center, label_text,
                            ha='center', va='center', color='black', fontsize=8)
        accumulations += weights

    if is_rate:
        if is_rotate:
            ax.set_ylim(0, 1)
        else:
            ax.set_xlim(0, 1)
    
    return ticklabels, keys


def plot_resistance_by_cluster_fixed(
    info: pd.DataFrame,
    smiles_name_dict: dict[str, dict[str, str]],
    key_col: str,
    cluster_col: str = "cluster_labels",
    *,
    sort_key: Sequence[Any] = [0],
    is_rate: bool = False,
    is_reverse: bool = False,
    is_rotate: bool = False,
    with_structures: bool = True,
    image_zoom: float = 0.25,
    label_x: float = -0.1,
    image_x_offset: float = 0.1,
    image_y_offset: float = 0.0
) -> dict[int, tuple[Figure, Axes]]:
    """
    Plot bar graphs of resistance distribution for each cluster.
    
    Parameters
    ----------
    info : pd.DataFrame
        DataFrame containing resistance data with cluster labels
    smiles_name_dict : dict
        Dictionary mapping keys to names and abbreviations
    key_col : str
        Column name for sample keys
    cluster_col : str, optional
        Column name for cluster labels, by default "cluster_labels"
    sort_key : Sequence[Any], optional
        Keys to sort by, by default [0]
    is_rate : bool, optional
        If True, show as rates, by default False
    is_reverse : bool, optional
        If True, reverse sort order, by default False
    is_rotate : bool, optional
        If True, rotate to vertical bars, by default False
    with_structures : bool, optional
        If True, add chemical structure images, by default True
    image_zoom : float, optional
        Zoom factor for images, by default 0.25
    label_x : float, optional
        X position for labels, by default -0.1
    image_x_offset : float, optional
        X offset for images, by default 0.1
    image_y_offset : float, optional
        Y offset for images, by default 0.0
    
    Returns
    -------
    dict[int, tuple[Figure, Axes]]
        Dictionary mapping cluster number to (Figure, Axes) tuple
    """
    results: dict[int, tuple[Figure, Axes]] = {}
    unique_clusters = sorted(info[cluster_col].unique())
    
    # For horizontal bar graphs, unify the axis limit based on the maximum count
    if not is_rate and not is_rotate:
        total_counts = info.groupby(key_col)["resistance_binary"].count()
        global_max = total_counts.max()
    else:
        global_max = 1.0
    
    common_ax_pos = [0.3, 0.15, 0.65, 0.8]
    
    for cluster in unique_clusters:
        cluster_data = info[info[cluster_col] == cluster]
        fig = plt.figure(figsize=(9.6, 6))
        ax = fig.add_axes(common_ax_pos)
        
        ticklabels, keys = _plot_resistance_binary_dists_fixed(
            cluster_data,
            smiles_name_dict,
            key_col,
            ax,
            sort_key=sort_key,
            is_rate=is_rate,
            is_reverse=is_reverse,
            is_rotate=is_rotate,
        )
        
        if not is_rate:
            cluster_count = cluster_data.shape[0]
            ax.set_title(f"Cluster {cluster} (n = {cluster_count})")
        else:
            ax.set_title(f"Cluster {cluster}")
        
        if not is_rotate and not is_rate:
            ax.set_xlim(0, global_max)
        
        if with_structures:
            add_structure_images(
                ax,
                keys,
                smiles_name_dict,
                get_rdkit_image,
                image_zoom=image_zoom,
                label_x=label_x,
                image_x_offset=image_x_offset,
                image_y_offset=image_y_offset
            )
            
        results[cluster] = (fig, ax)
    
    return results


def draw_binary_distribution(
    info: pd.DataFrame,
    smiles_name_dict: dict[str, dict[str, str]],
    key_col: str,
    ax: Axes,
    *,
    sort_key: Sequence[Any] = [0, 1],
    is_rate: bool = False,
    is_reverse: bool = False,
    is_rotate: bool = False,
    show_ylabels: bool = True,
    font_size: int = None
) -> list[str]:
    """
    Draw binary resistance distribution on the specified Axes.
    
    Parameters
    ----------
    info : pd.DataFrame
        DataFrame containing resistance data
    smiles_name_dict : dict
        Dictionary mapping keys to names and abbreviations
    key_col : str
        Column name for sample keys
    ax : Axes
        Matplotlib Axes to plot on
    sort_key : Sequence[Any], optional
        Keys to sort by, by default [0, 1]
    is_rate : bool, optional
        If True, show as rates, by default False
    is_reverse : bool, optional
        If True, reverse sort order, by default False
    is_rotate : bool, optional
        If True, rotate to vertical bars, by default False
    show_ylabels : bool, optional
        If True, show y-axis labels, by default True
    
    Returns
    -------
    list[str]
        List of tick labels used in the bar graph
    """
    # Extract samples using the key and generate abbreviations (tick labels)
    key_samples = info.drop_duplicates(subset=key_col)
    keys: list[str] = key_samples[key_col].tolist()
    ticklabels = [smiles_name_dict[key]["jp_abbr"] for key in keys]

    # Create a DataFrame of resistance_binary counts for each key
    counts = pd.DataFrame({
        ylabel: info.loc[info[key_col] == key, "resistance_binary"].value_counts()
        for key, ylabel in zip(keys, ticklabels, strict=True)
    }).fillna(0).astype(int)

    # Convert to percentages and sort based on the provided sort_key
    rate_values = counts.div(counts.sum(axis=0), axis=1)
    sorted_columns = rate_values.sort_values(by=sort_key, ascending=not is_reverse, axis=1).columns.tolist()

    values = counts if not is_rate else rate_values
    values = values[sorted_columns]
    binary_ticklabels = values.columns.tolist()
    objs = values.index.tolist()

    colors = [COLOR_RESISTANT, COLOR_NON_RESISTANT]
    accumulations = pd.Series(0, index=binary_ticklabels)

    for i, obj in enumerate(objs):
        weights = values.loc[obj]
        if is_rotate:
            ax.bar(
                values.columns,
                weights,
                bottom=accumulations,
                label=str(obj),
                color=colors[i % len(colors)],
                alpha=ALPHA_HIST,
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
        else:
            ax.barh(
                values.columns,
                weights,
                left=accumulations,
                label=str(obj),
                color=colors[i % len(colors)],
                alpha=ALPHA_HIST,
            )
        accumulations += weights

    if is_rate:
        if is_rotate:
            ax.set_ylim(0, 1)
        else:
            ax.set_xlim(0, 1)
    else:
        if is_rotate:
            ax.set_ylim(0, 100)
        else:
            ax.set_xlim(0, 100)

    if not show_ylabels:
        if is_rotate:
            ax.set_xticklabels([])
        else:
            ax.set_yticklabels([])

    # Apply font sizes (use provided font_size or defaults)
    if font_size is not None:
        label_size = int(font_size * 1.125)
        title_size = int(font_size * 1.25)
        tick_size = font_size
    else:
        label_size = 14
        title_size = 16
        tick_size = 12
    
    ax.set_xlabel("Resistance Binary Distribution", fontsize=label_size)
    ax.set_title("Resistance Binary Distribution", fontsize=title_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)

    return binary_ticklabels


def draw_percentage_distribution(
    info: pd.DataFrame,
    key_col: str,
    ax: Axes,
    *,
    sort_key: Sequence[Any] = ['resistance_0_(%)', 'resistance_1_(%)'],
    is_rate: bool = False,
    show_ylabels: bool = True,
    font_size: int = None
) -> None:
    """
    Draw resistance percentage distribution on the specified Axes.
    
    Parameters
    ----------
    info : pd.DataFrame
        DataFrame containing resistance percentage data
    key_col : str
        Column name for sample keys
    ax : Axes
        Matplotlib Axes to plot on
    sort_key : Sequence[Any], optional
        Keys to sort by, by default ['resistance_0_(%)', 'resistance_1_(%)']
    is_rate : bool, optional
        If True, show as rates (0-1), by default False
    show_ylabels : bool, optional
        If True, show y-axis labels, by default True
    """
    # For each key, calculate the mean of the target columns
    counts = pd.DataFrame({
        key: info.loc[info[key_col] == key, ['resistance_0_(%)', 'resistance_1_(%)']].mean()
        for key in info[key_col].unique()
    })
    values = counts.fillna(0).astype(int)
    if is_rate:
        values = values.div(values.sum(axis=0), axis=1)

    values = values.sort_values(by=sort_key, ascending=False, axis=1)
    percent_ticklabels: list[str] = values.columns.tolist()
    objs = values.index.tolist()

    colors = [COLOR_RESISTANT, COLOR_NON_RESISTANT]
    accumulations = pd.Series(0, index=percent_ticklabels)

    for i, obj in enumerate(objs):
        weights = values.loc[obj]
        ax.barh(
            values.columns,
            weights,
            left=accumulations,
            label=str(obj),
            color=colors[i % len(colors)],
            alpha=ALPHA_HIST,
        )
        accumulations += weights

    ax.invert_yaxis()

    if is_rate:
        ax.set_xlim(0, 1)
    else:
        ax.set_xlim(0, 100)

    yticks = range(1, len(percent_ticklabels) + 1)
    ax.set_yticks(yticks)
    if show_ylabels:
        ax.set_yticklabels(percent_ticklabels[::-1])
    else:
        ax.set_yticklabels([])

    # Apply font sizes (use provided font_size or defaults)
    if font_size is not None:
        label_size = int(font_size * 1.125)
        title_size = int(font_size * 1.25)
        tick_size = font_size
    else:
        label_size = 14
        title_size = 16
        tick_size = 12
    
    ax.set_xlabel("Resistance Percentage Distribution", fontsize=label_size)
    ax.set_title("Resistance Percentage Distribution", fontsize=title_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)


def plot_combined_resistance_distributions(
    binary_info: pd.DataFrame,
    percentage_info: pd.DataFrame,
    smiles_name_dict: dict[str, dict[str, str]],
    key_col_binary: str,
    key_col_percentage: str,
    *,
    is_rate: bool = True,
    binary_is_reverse: bool = True,
    binary_is_rotate: bool = False,
    show_ylabels: bool = True
) -> Figure:
    """
    Create combined plot with binary and percentage resistance distributions.
    
    Parameters
    ----------
    binary_info : pd.DataFrame
        DataFrame containing binary resistance data
    percentage_info : pd.DataFrame
        DataFrame containing percentage resistance data
    smiles_name_dict : dict
        Dictionary mapping keys to names and abbreviations
    key_col_binary : str
        Column name for binary data keys
    key_col_percentage : str
        Column name for percentage data keys
    is_rate : bool, optional
        If True, show as rates, by default True
    binary_is_reverse : bool, optional
        If True, reverse sort order for binary data, by default True
    binary_is_rotate : bool, optional
        If True, rotate binary bars to vertical, by default False
    show_ylabels : bool, optional
        If True, show y-axis labels, by default True
    
    Returns
    -------
    Figure
        Matplotlib Figure object
    """
    fig, (ax_top, ax_bottom) = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
    
    # Top: Draw the binary resistance distribution graph
    draw_binary_distribution(
        info=binary_info,
        smiles_name_dict=smiles_name_dict,
        key_col=key_col_binary,
        ax=ax_top,
        sort_key=[0, 1],
        is_rate=is_rate,
        is_reverse=binary_is_reverse,
        is_rotate=binary_is_rotate,
        show_ylabels=show_ylabels
    )
    
    # Bottom: Draw the resistance percentage distribution graph
    draw_percentage_distribution(
        info=percentage_info,
        key_col=key_col_percentage,
        ax=ax_bottom,
        sort_key=['resistance_0_(%)', 'resistance_1_(%)'],
        is_rate=is_rate,
        show_ylabels=show_ylabels
    )
    
    # Unify the horizontal axis range for both graphs
    ax_top.set_xlim(0, 1)
    ax_bottom.set_xlim(0, 1)
    
    plt.tight_layout()
    
    return fig

# =============================================================================
# FIGURE GENERATION FUNCTIONS FOR PAPER
# =============================================================================

def generate_figure1_heatmap(
    valid_info_cluster: pd.DataFrame,
    save_dir: str = None
) -> Figure:
    """
    Generate Figure 1: Chemical Resistance Heatmap (Main Figure).
    
    This function creates a heatmap showing the chemical resistance matrix
    of polymer-solvent combinations.
    
    Parameters
    ----------
    valid_info_cluster : pd.DataFrame
        DataFrame containing resistance data with columns:
        - 'solvent': solvent names
        - 'symbol': polymer symbols
        - 'resistance_binary': binary resistance values (0/1)
        - 'solvent_rank': ranking for sorting solvents
        - 'polymer_rank': ranking for sorting polymers
    save_dir : str, optional
        Directory to save the figure (PDF and SVG formats)
    
    Returns
    -------
    Figure
        Matplotlib Figure object containing the heatmap
    """
    import matplotlib as mpl
    
    # Set font parameters for publication quality
    mpl.rcParams['pdf.fonttype'] = 42  # Save font as TrueType
    mpl.rcParams['font.family'] = 'Arial'
    
    # Sort data by rank
    df_sorted = valid_info_cluster.sort_values(['solvent_rank', 'polymer_rank'])
    
    # Create pivot table
    heatmap_data = df_sorted.pivot(
        index='solvent', 
        columns='symbol', 
        values='resistance_binary'
    )
    
    # Get sorted orders
    solvent_order = df_sorted.sort_values('solvent_rank')['solvent'].unique()
    symbol_order = df_sorted.sort_values('polymer_rank')['symbol'].unique()
    
    # Reorder heatmap data
    heatmap_data = heatmap_data.loc[solvent_order, symbol_order]
    
    # Set custom colormap (0: Green, 1: Red)
    cmap = sns.color_palette([COLOR_RESISTANT, COLOR_NON_RESISTANT])
    
    # Cell border parameters
    cell_linewidth = 0.25
    external_linewidth = 10
    
    # Create figure (ACS Macromolecules 2-column layout: 7x6 inches)
    fig = plt.figure(figsize=(7, 6))
    
    ax = sns.heatmap(
        heatmap_data,
        cmap=cmap,
        cbar=False,
        annot=False,
        alpha=0.5,
        linecolor='black',
        linewidths=cell_linewidth
    )
    
    # Set title
    total_count = len(valid_info_cluster)
    n_solvents = len(solvent_order)
    n_polymers = len(symbol_order)
    plt.title(
        f"Chemical resistance dataset\n"
        f"total number : {total_count:,}, solvent : {n_solvents}, polymer : {n_polymers}",
        fontsize=14
    )
    
    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # X-axis settings
    x_tick_fontsize = 8
    num_symbols = len(symbol_order)
    ax.set_xticks(np.arange(num_symbols) + 0.5)
    ax.set_xticklabels(symbol_order, rotation=90, fontsize=x_tick_fontsize)
    
    # Y-axis settings
    num_solvents = len(solvent_order)
    if num_solvents > 50:
        # If too many solvents, show only 50 labels
        tick_indices = np.linspace(0, num_solvents - 1, 50, dtype=int)
        selected_labels = [solvent_order[i] for i in tick_indices]
        tick_locs = np.linspace(0.5, num_solvents - 0.5, 50)
        ax.set_yticks(tick_locs)
        ax.set_yticklabels(selected_labels)
    else:
        ax.set_yticks(np.arange(num_solvents) + 0.5)
        ax.set_yticklabels(solvent_order)
    
    y_tick_fontsize = 8
    plt.yticks(rotation=0)
    plt.setp(ax.get_yticklabels(), fontsize=y_tick_fontsize)
    
    # Highlight specific polymer labels in red
    special_labels = ['PVA', 'PIB', 'PMMA', 'PVdC', 'PC', 'PVC', 'PS', 'PSF', 'PMP']
    for label in ax.get_xticklabels():
        if label.get_text() in special_labels:
            label.set_color('red')
    
    # Hide tick marks (keep labels)
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
    )
    
    # Set outer frame thickness
    for spine in ax.spines.values():
        spine.set_linewidth(external_linewidth)
    
    plt.tight_layout()
    
    # Save if directory is provided
    if save_dir:
        import os
        output_path_pdf = os.path.join(save_dir, 'fig1.pdf')
        output_path_svg = os.path.join(save_dir, 'fig1.svg')
        output_path_jpg = os.path.join(save_dir, 'fig1.jpg')
        plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
        plt.savefig(output_path_svg, format='svg', bbox_inches='tight')
        plt.savefig(output_path_jpg, format='jpg', dpi=300, bbox_inches='tight')
        print(f"Figure 1 saved to {output_path_pdf} and {output_path_svg}")
    
    return fig


def generate_figure1_cluster_heatmaps(
    valid_info_cluster: pd.DataFrame,
    df_sorted: pd.DataFrame,
    solvent_name_dict: dict,
    save_dir: str = None
) -> dict:
    """
    Generate cluster-wise heatmaps with chemical structure images.
    
    Parameters
    ----------
    valid_info_cluster : pd.DataFrame
        DataFrame containing resistance data with cluster labels
    df_sorted : pd.DataFrame
        Sorted DataFrame for determining polymer order
    solvent_name_dict : dict
        Dictionary mapping SMILES to solvent names/abbreviations
    save_dir : str, optional
        Directory to save the figures
    
    Returns
    -------
    dict
        Dictionary mapping cluster number to Figure object
    """
    # Increase font size
    plt.rcParams.update({'font.size': 16})
    
    # Polymer order from df_sorted
    symbol_order = df_sorted.sort_values('polymer_rank')['symbol'].unique()
    
    # Special labels to highlight
    special_labels = ['PVA', 'PIB', 'PMMA', 'PVdC', 'PC', 'PVC', 'PS', 'PSF', 'PMP']
    
    # Process each cluster
    unique_clusters = sorted(valid_info_cluster["cluster_labels"].unique())
    figures = {}
    
    for cluster in unique_clusters:
        # Extract cluster data
        cluster_data = valid_info_cluster[
            valid_info_cluster["cluster_labels"] == cluster
        ].copy()
        cluster_data.sort_values(['smiles_solvent'], inplace=True)
        
        # Create pivot table
        pivot_data = cluster_data.pivot(
            index='smiles_solvent',
            columns='symbol',
            values='resistance_binary'
        )
        
        # Reorder
        smiles_order = cluster_data['smiles_solvent'].unique()
        pivot_data = pivot_data.reindex(index=smiles_order)
        pivot_data = pivot_data.reindex(columns=symbol_order)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        cmap = ListedColormap([(0, 0.5, 0, 0.5), (1, 0.7, 0.7, 0.5)])
        sns.heatmap(
            pivot_data,
            ax=ax,
            cmap=cmap,
            cbar=False,
            annot=False,
            linewidths=0.1,
            linecolor='gray',
            vmin=0,
            vmax=1
        )
        
        # Set title
        ax.set_title(
            f'Cluster {cluster}, Data count: {len(cluster_data)}',
            fontsize=20
        )
        
        # Remove axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        
        # Highlight special labels
        for label in ax.get_xticklabels():
            if label.get_text() in special_labels:
                label.set_color('red')
        
        # Add chemical structure images
        solvent_smiles_keys = list(pivot_data.index)
        add_structure_images(
            ax,
            solvent_smiles_keys,
            solvent_name_dict,
            get_rdkit_image,
            image_zoom=0.3,
            label_x=-0.13,
            image_x_offset=0.05,
            image_y_offset=0.0
        )
        
        fig.tight_layout(pad=2.0)
        
        # Save if directory provided
        if save_dir:
            import os
            svg_path = os.path.join(save_dir, f"resistance_matrix_cluster_{cluster}.svg")
            jpg_path = os.path.join(save_dir, f"resistance_matrix_cluster_{cluster}.jpg")
            save_as_png(fig, svg_path)
            save_as_png(fig, jpg_path)
        
        figures[cluster] = fig
    
    return figures

def draw_polymer_graph_for_fig3(
    ax: 'Axes',
    dfs: dict,
    values: pd.Series,
    valid_info: pd.DataFrame,
    polymer_name_dict: dict,
    base_font_size: int = 40
) -> None:
    """
    Draw polymer AUC score graph for Figure 3.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib Axes object
    dfs : dict
        Dictionary containing 'auc' DataFrame
    values : pd.Series
        Series containing values to plot
    valid_info : pd.DataFrame
        DataFrame with resistance information
    polymer_name_dict : dict
        Dictionary mapping SMILES to polymer names
    base_font_size : int
        Base font size for labels
    """
    import re
    
    title_font_size = int(base_font_size * 1.25)
    label_font_size = int(base_font_size * 1.125)
    tick_font_size = base_font_size
    annotation_font_size = base_font_size - 2
    
    df_opt = dfs["auc"]
    df_polymer = df_opt.iloc[0:1, 0:39]
    df_polymer = df_polymer.sort_values(by=df_polymer.index[0], axis=1, ascending=False)
    
    polymer_list = [
        re.sub(r'^cv_|_(micro_f1|macro_f1|auc|accuracy|fpr|fnr)$', '', col)
        for col in df_polymer.columns
    ]
    df_new = df_polymer.copy()
    df_new.columns = polymer_list
    
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
    
    ax.set_xlabel('Leave one group out CV AUC values', fontsize=label_font_size)
    ax.set_title('2 class classification LOGOCV AUC score for polymer', fontsize=title_font_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
    
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
                va='center', fontsize=annotation_font_size)
    
    ax.text(mean_value - 0.15, ax.get_ylim()[1]-0.1, f'Mean AUC= {mean_value:.2f}',
            va='top', ha='center', color='red', fontsize=tick_font_size)


def draw_solvent_graph_for_fig3(
    ax: 'Axes',
    dfs: dict,
    values: pd.Series,
    base_font_size: int = 40
) -> None:
    """
    Draw solvent cluster AUC score graph for Figure 3.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib Axes object
    dfs : dict
        Dictionary containing 'auc' DataFrame
    values : pd.Series
        Series containing values to plot (not used, kept for consistency)
    base_font_size : int
        Base font size for labels
    """
    import re
    
    title_font_size = int(base_font_size * 1.25)
    label_font_size = int(base_font_size * 1.125)
    tick_font_size = base_font_size
    annotation_font_size = base_font_size - 2
    
    df_opt_solvent = dfs["auc"]
    df_solvent = df_opt_solvent.loc[:, df_opt_solvent.columns.str.startswith('cv_solv_cluster')]
    df_solvent = df_solvent.sort_index(
        axis=1, 
        key=lambda x: x.str.extract(r'(\d+)').astype(int)[0],
        ascending=True
    )
    
    # Rename columns: 'cv_solv_cluster_X_auc' -> 'cluster_X'
    solv_cluster_list = [re.sub(r'^cv_solv_cluster', 'cluster', col) for col in df_solvent.columns]
    solv_cluster_list = [re.sub(r'_(micro_f1|macro_f1|auc|accuracy|fpr|fnr)$', '', col) for col in solv_cluster_list]
    df_solvent.columns = solv_cluster_list
    
    # Reverse order for display
    df_new_solvent = df_solvent[df_solvent.columns[::-1]]
    objs_solvent = df_new_solvent.columns.tolist()
    values_solvent = df_new_solvent.iloc[0]
    color = sns.color_palette('deep')[0]
    
    bars_solvent = ax.barh(objs_solvent, values_solvent, color=color, alpha=0.5)
    mean_value_solvent = values_solvent.mean()
    ax.axvline(mean_value_solvent, color='red', linestyle='dashed')
    
    ax.set_xlabel('Leave one group out CV AUC values', fontsize=label_font_size)
    ax.set_title('2 class classification LOGOCV AUC score for solvent', fontsize=title_font_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
    
    for bar, val in zip(bars_solvent, values_solvent):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                va='center', fontsize=annotation_font_size)
    
    ax.text(mean_value_solvent - 0.15, ax.get_ylim()[1]-0.1, f'Mean AUC = {mean_value_solvent:.2f}',
            va='top', ha='center', color='red', fontsize=tick_font_size)


def generate_figure3_combined(
    dfs: dict,
    values: pd.Series,
    valid_info: pd.DataFrame,
    valid_info_cluster: pd.DataFrame,
    average_df: pd.DataFrame,
    polymer_name_dict: dict,
    save_dir: str = None
) -> Figure:
    """
    Generate Figure 3: Feature Distribution Analysis (Combined 4-panel figure).
    
    This function creates a 2x2 grid of subplots showing:
    (a) Polymer AUC scores
    (b) Solvent cluster AUC scores
    (c) Binary resistance distribution
    (d) Percentage distribution by cluster
    
    Parameters
    ----------
    dfs : dict
        Dictionary containing DataFrames (must have 'auc' key)
    values : pd.Series
        Series with metric values
    valid_info : pd.DataFrame
        DataFrame with validation information
    valid_info_cluster : pd.DataFrame
        DataFrame with clustered validation information
    average_df : pd.DataFrame
        DataFrame with averaged resistance percentages
    polymer_name_dict : dict
        Dictionary mapping SMILES to polymer names
    save_dir : str, optional
        Directory to save the figure
    
    Returns
    -------
    Figure
        Matplotlib Figure object
    """
    # Font settings
    base_font_size = 40
    title_font_size = int(base_font_size * 1.25)
    
    plt.rcParams.update({
        'font.size': base_font_size,
        'axes.titlesize': title_font_size,
        'axes.labelsize': int(base_font_size * 1.125),
        'xtick.labelsize': base_font_size,
        'ytick.labelsize': base_font_size,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'pdf.fonttype': 42
    })
    
    # Create figure with 2x2 subplots (matched to original layout)
    # Original figsize was (44, 32)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(44, 32))
    
    ax_polymer = axs[0, 0]
    ax_solvent = axs[1, 0]
    ax_binary = axs[0, 1]
    ax_percentage = axs[1, 1]
    
    # Draw each subplot
    draw_polymer_graph_for_fig3(ax_polymer, dfs, values, valid_info, polymer_name_dict, base_font_size)
    draw_solvent_graph_for_fig3(ax_solvent, dfs, values, base_font_size)
    ax_polymer.set_xlim(0, 1.05)
    ax_solvent.set_xlim(0, 1.05)
    
    draw_binary_distribution(
        info=valid_info_cluster,
        smiles_name_dict=polymer_name_dict,
        key_col="smiles_polymer",
        ax=ax_binary,
        sort_key=[0, 1],
        is_rate=True,
        is_reverse=True,
        is_rotate=False,
        show_ylabels=True,
        font_size=base_font_size
    )
    draw_percentage_distribution(
        info=average_df,
        key_col="cluster_labels",
        ax=ax_percentage,
        sort_key=['resistance_0_(%)', 'resistance_1_(%)'],
        is_rate=True,
        show_ylabels=True,
        font_size=base_font_size
    )
    ax_binary.set_xlim(0, 1)
    ax_percentage.set_xlim(0, 1)
    
    # Hide unnecessary y-tick labels
    for ax in [ax_binary, ax_percentage]:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', length=0)
    
    # Add subplot labels (a), (b), (c), (d) - matching original layout
    # Polymer (top-left), Binary (top-right), Solvent (bottom-left), Percentage (bottom-right)
    for ax_obj, label in zip([ax_polymer, ax_binary, ax_solvent, ax_percentage], 
                              ["(a)", "(c)", "(b)", "(d)"]):
        ax_obj.text(
            -0.05, 1.05, label,
            transform=ax_obj.transAxes,
            fontsize=title_font_size,
            fontweight="normal",
            fontfamily="Arial",
            va="top", ha="left",
            clip_on=False
        )
    
    plt.tight_layout()
    
    # Save if directory provided
    if save_dir:
        import os
        output_path_svg = os.path.join(save_dir, "fig3.svg")
        output_path_jpg = os.path.join(save_dir, "fig3.jpg")
        output_path_pdf = os.path.join(save_dir, "fig3.pdf")
        plt.savefig(output_path_svg, dpi=300)
        plt.savefig(output_path_jpg, dpi=300)
        plt.savefig(output_path_pdf, format="pdf", dpi=300)
        print(f"Figure 3 saved to {output_path_svg} and {output_path_pdf}")
    
    return fig


def generate_figure4_mic_and_rg(
    mic_csv_path: str,
    df_acetone_vPolymer: pd.DataFrame,
    save_dir: str = None,
    base_fontsize: int = 12
) -> Figure:
    """
    Generate Figure 4: Feature Importance Analysis (MIC + Rg Distribution).
    
    This function creates a 2-panel figure showing:
    (a) MIC (Maximal Information Coefficient) bar graph for crystallinity features
    (b) Histogram of radius of gyration (Rg) distribution by crystallinity
    
    Parameters
    ----------
    mic_csv_path : str
        Path to CSV file containing MIC scores
    df_acetone_vPolymer : pd.DataFrame
        DataFrame containing acetone-polymer data with Rg and crystallinity info
    save_dir : str, optional
        Directory to save the figure (SVG and PDF formats)
    base_fontsize : int, optional
        Base font size for labels (default: 12)
    
    Returns
    -------
    Figure
        Matplotlib Figure object
    """
    import os
    
    # =============================
    # (1) Left graph: MIC bar graph
    # =============================
    
    # Read data from CSV file
    df_mic = pd.read_csv(mic_csv_path)
    
    # Step 1: Extract only rows not starting with 'FF_'
    df_mic_filtered = df_mic[~df_mic['feature'].str.startswith('FF_')].copy()
    
    # Step 2: Remove '_radonpy_polymer' (for display)
    df_mic_filtered['feature_clean'] = df_mic_filtered['feature'].str.replace(
        '_radonpy_polymer', '', regex=False
    )
    
    # MD column name mapping dictionary
    MD_column_dict = {
        "density": "Density",
        "Rg": "Radius of gyration (Rg)",
        "Scaled Rg": "Scaled radius of gyration",
        "self-diffusion": "Self-diffusion coefficient",
        "Cp": "Heat capacity at constant pressure (Cp)",
        "Cv": "Heat capacity at constant volume (Cv)",
        "compressibility": "Compressibility",
        "isentropic_compressibility": "Isentropic compressibility",
        "bulk_modulus": "Bulk modulus",
        "isentropic_bulk_modulus": "Isentropic bulk modulus",
        "volume_expansion": "Volumetric expansion coefficient", 
        "linear_expansion": "Linear expansion coefficient", 
        "r2": "End-to-end distance", 
        "static_dielectric_const": "Static dielectric constant", 
        "dielectric_const_dc": "Dielectric constant", 
        "nematic_order_parameter": "Nematic order parameter", 
        "refractive_index": "Refractive index"
    }
    
    # Rename feature_clean column using dictionary
    df_mic_filtered['feature_clean'] = df_mic_filtered['feature_clean']\
        .map(MD_column_dict).fillna(df_mic_filtered['feature_clean'])
    
    # Get top 18 items
    df_top10 = df_mic_filtered.sort_values(by='MIC', ascending=False).head(18)
    
    # =============================
    # (2) Right graph: Histogram
    # =============================
    
    # Target columns
    chi_col = "Rg_radonpy_polymer"
    binary_col = "crystalinity_binary"
    
    # Extract data
    chi = df_acetone_vPolymer[chi_col]
    binary = df_acetone_vPolymer[binary_col]
    
    # Split data by binary label
    chi_0 = chi[binary == 0]
    chi_1 = chi[binary == 1]
    
    # Count number of data points
    n_0 = len(chi_0)
    n_1 = len(chi_1)
    
    # Setting x-axis range and bins
    xlim = (10, 60)
    bins = np.linspace(xlim[0], xlim[1], 51)
    
    # =============================
    # (3) Draw two graphs side by side
    # =============================
    
    # Create subplots: 2 columns
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 10))
    
    # ----- ax1: Left bar graph (MIC) -----
    bars = ax1.bar(df_top10['feature_clean'], df_top10['MIC'], color='skyblue')
    
    # Axis settings
    ax1.set_xticklabels(
        df_top10['feature_clean'], 
        rotation=45, 
        ha='right', 
        fontsize=base_fontsize
    )
    ax1.set_ylabel("MIC", fontsize=base_fontsize)
    ax1.set_title(
        "MIC of crystallinity prediction probability", 
        fontsize=base_fontsize+4
    )
    ax1.set_ylim(0, 0.7)
    ax1.tick_params(axis='both', labelsize=base_fontsize)
    
    # ----- ax2: Right histogram -----
    ax2.hist(
        chi_0, bins=bins, color="gray", alpha=0.5, 
        label=f"Amorphous (n = {n_0:,})", density=True
    )
    ax2.hist(
        chi_1, bins=bins, color="lightblue", alpha=0.5, 
        label=f"Crystalline (n = {n_1:,})", density=True
    )
    
    ax2.set_xlabel("Radius of gyration (Rg)", fontsize=base_fontsize)
    ax2.set_ylabel("Relative frequency", fontsize=base_fontsize)
    ax2.set_title(
        "Distribution of radius of gyration (Rg)", 
        fontsize=base_fontsize+4
    )
    ax2.legend(fontsize=base_fontsize)
    ax2.set_xlim(xlim)
    ax2.tick_params(axis='both', labelsize=base_fontsize)
    
    # ----- Add (a) and (b) labels -----
    # Get subplot position information
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    
    # Add labels in the top-left corner of each subplot
    fig.text(
        pos1.x0 - 0.07, pos1.y1 + 0.04, "(a)",
        transform=fig.transFigure,
        fontsize=base_fontsize+4,
        fontweight="normal",
        fontfamily="Arial",
        ha="left", va="bottom"
    )
    fig.text(
        pos2.x0 - 0.03, pos2.y1 + 0.04, "(b)",
        transform=fig.transFigure,
        fontsize=base_fontsize+4,
        fontweight="normal",
        fontfamily="Arial",
        ha="left", va="bottom"
    )
    
    # Layout adjustment
    plt.tight_layout()
    
    # Save if directory provided
    if save_dir:
        filename_base = os.path.join(save_dir, "fig4")
        plt.savefig(filename_base + ".svg", format="svg", dpi=300, bbox_inches="tight")
        plt.savefig(filename_base + ".jpg", format="jpg", dpi=300, bbox_inches="tight")
        plt.savefig(filename_base + ".pdf", format="pdf", dpi=300, bbox_inches="tight")
        print(f"Figure 4 saved to {filename_base}.svg, .jpg and .pdf")
    
    return fig


def generate_figure5_resistance_prediction(
    df_all_mic: pd.DataFrame,
    df_crystal_mic: pd.DataFrame,
    df_noncrystal_mic: pd.DataFrame,
    pred_CL0: pd.Series,
    pred_CL1: pd.Series,
    n_all: int,
    save_dir: str = None
) -> Figure:
    """
    Generate Figure 5: Resistance Prediction Results.
    
    This function creates a 2-panel figure showing:
    (a) MIC bar graph comparing All, Crystalline, and Amorphous data
    (b) Histogram of non-resistance probability by crystallinity
    
    Parameters
    ----------
    df_all_mic : pd.DataFrame
        DataFrame with MIC scores for all data
    df_crystal_mic : pd.DataFrame
        DataFrame with MIC scores for crystalline data
    df_noncrystal_mic : pd.DataFrame
        DataFrame with MIC scores for amorphous data
    pred_CL0 : pd.Series
        Resistance predictions for amorphous polymers
    pred_CL1 : pd.Series
        Resistance predictions for crystalline polymers
    n_all : int
        Total number of data points
    save_dir : str, optional
        Directory to save the figure
    
    Returns
    -------
    Figure
        Matplotlib Figure object
    """
    plt.rcParams.update({'font.size': 24})
    
    # MD column name mapping dictionary
    MD_column_dict = {
        "crystalinity_binary": "Crystallinity",
        "density": "Density",
        "rg": "Radius of gyration (Rg)",
        "scaled rg": "Scaled radius of gyration",
        "self-diffusion": "Self-diffusion coefficient",
        "cp": "Heat capacity at constant pressure (Cp)",
        "cv": "Heat capacity at constant volume (Cv)",
        "compressibility": "Compressibility",
        "isentropic_compressibility": "Isentropic compressibility",
        "bulk_modulus": "Bulk modulus",
        "isentropic_bulk_modulus": "Isentropic bulk modulus",
        "volume_expansion": "Volumetric expansion coefficient", 
        "linear_expansion": "Linear expansion coefficient", 
        "r2": "End-to-end distance", 
        "static_dielectric_const": "Static dielectric constant", 
        "dielectric_const_dc": "Dielectric constant", 
        "nematic_order_parameter": "Nematic order parameter", 
        "refractive_index": "Refractive index"
    }
    
    top_n = 5
    n_CL0 = len(pred_CL0)
    n_CL1 = len(pred_CL1)
    
    # Extract top features with high MIC from df_all
    top_features = df_all_mic.sort_values(by='MIC', ascending=False).head(top_n)['feature']
    
    # Convert to lowercase and map to clean names
    top_features_clean = [
        feature.replace('_radonpy_polymer', '').lower() 
        for feature in top_features
    ]
    top_features_mapped = [
        MD_column_dict.get(feature, feature) 
        for feature in top_features_clean
    ]
    
    def get_mic_values_subset(df, features):
        df_sub = df[df['feature'].isin(features)].copy()
        return df_sub.set_index('feature')['MIC'].reindex(features).values
    
    mic_all = get_mic_values_subset(df_all_mic, top_features)
    mic_crystal = get_mic_values_subset(df_crystal_mic, top_features)
    mic_noncrystal = get_mic_values_subset(df_noncrystal_mic, top_features)
    
    x = np.arange(top_n)
    bar_width = 0.25
    
    # Create 2-panel figure
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(24, 12))
    
    # Left side: MIC bar graph
    ax1.bar(
        x - bar_width, mic_all, width=bar_width,
        label=f'All (n = {n_all:,})', color='orange'
    )
    ax1.bar(
        x, mic_crystal, width=bar_width,
        label=f'Crystalline (n = {n_CL1:,})', color='lightblue', alpha=0.8
    )
    ax1.bar(
        x + bar_width, mic_noncrystal, width=bar_width,
        label=f'Amorphous (n = {n_CL0:,})', color='gray', alpha=0.5
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_features_mapped, rotation=45, ha='center', fontsize=20)
    ax1.set_ylabel("MIC", fontsize=24)
    ax1.legend(fontsize=20)
    ax1.tick_params(axis='both', labelsize=20)
    ax1.set_title("MIC of resistance prediction probability", fontsize=28)
    
    # Right side: Histogram (Resistance probability)
    ax2.hist(
        pred_CL1, bins=20, alpha=0.8, color='lightblue', density=True,
        label=f'Crystalline (n = {n_CL1:,})'
    )
    ax2.hist(
        pred_CL0, bins=20, alpha=0.5, color='gray', density=True,
        label=f'Amorphous (n = {n_CL0:,})'
    )
    ax2.set_xlabel("Non-resistance probability", fontsize=24)
    ax2.set_ylabel("Normalized frequency", fontsize=24)
    ax2.tick_params(axis='both', labelsize=20)
    ax2.legend(fontsize=20)
    ax2.set_title("Histogram of non-resistance probability", fontsize=28)
    
    # Add subplot labels (a) and (b)
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    fig.text(
        pos1.x0 - 0.11, pos1.y1 + 0.06, "(a)",
        transform=fig.transFigure,
        fontsize=28,
        fontweight="normal",
        fontfamily="Arial",
        ha="left", va="bottom"
    )
    fig.text(
        pos2.x0 - 0.04, pos2.y1 + 0.06, "(b)",
        transform=fig.transFigure,
        fontsize=28,
        fontweight="normal",
        fontfamily="Arial",
        ha="left", va="bottom"
    )
    
    plt.tight_layout()
    
    # Save if directory provided
    if save_dir:
        import os
        filename_base = os.path.join(save_dir, "fig5")
        plt.savefig(filename_base + ".svg", format="svg", dpi=300, bbox_inches="tight")
        plt.savefig(filename_base + ".jpg", format="jpg", dpi=300, bbox_inches="tight")
        plt.savefig(filename_base + ".pdf", format="pdf", dpi=300, bbox_inches="tight")
        print(f"Figure 5 saved to {filename_base}.svg, .jpg and .pdf")
    
    return fig


def generate_figure6_crystallinity_effect(
    df_acetone_vPolymer: pd.DataFrame,
    x_col: str = "density_radonpy_polymer",
    y_col: str = "resistance_pred",
    save_dir: str = None,
    base_fontsize: int = 20
) -> Figure:
    """
    Generate Figure 6: Crystallinity Effect on Resistance.
    
    This function creates a complex multi-panel figure showing:
    - 2D heatmaps of density vs resistance probability (crystalline and amorphous)
    - Marginal histograms for density (top) and resistance probability (left)
    
    Parameters
    ----------
    df_acetone_vPolymer : pd.DataFrame
        DataFrame containing acetone-polymer data with density, resistance prediction,
        and crystallinity information
    x_col : str, optional
        Column name for x-axis (density)
    y_col : str, optional
        Column name for y-axis (resistance prediction probability)
    save_dir : str, optional
        Directory to save the figure
    base_fontsize : int, optional
        Base font size for labels
    
    Returns
    -------
    Figure
        Matplotlib Figure object
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as mcolors
    
    # Font size settings
    title_fontsize = base_fontsize * 1.0
    label_fontsize = base_fontsize * 0.5
    tick_fontsize = base_fontsize * 0.45
    
    # Data splitting: amorphous (0) and crystalline (1)
    df_noncrystal = df_acetone_vPolymer[df_acetone_vPolymer["crystalinity_binary"] == 0]
    df_crystal = df_acetone_vPolymer[df_acetone_vPolymer["crystalinity_binary"] == 1]
    
    # Bin settings for histograms
    x_bins = np.linspace(0.8, 1.5, 21)
    y_bins = np.arange(0, 1.05, 0.05)
    dx = x_bins[1] - x_bins[0]
    dy = y_bins[1] - y_bins[0]
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2
    
    # Create 2D histograms for heatmaps
    heat_noncrystal, _, _ = np.histogram2d(
        df_noncrystal[x_col], df_noncrystal[y_col], bins=[x_bins, y_bins]
    )
    heat_crystal, _, _ = np.histogram2d(
        df_crystal[x_col], df_crystal[y_col], bins=[x_bins, y_bins]
    )
    
    # Normalize and transpose for imshow
    heat_noncrystal_norm = (heat_noncrystal / np.sum(heat_noncrystal)).T
    heat_crystal_norm = (heat_crystal / np.sum(heat_crystal)).T
    
    # Create 1D histograms
    hist_x_noncrystal, _ = np.histogram(df_noncrystal[x_col], bins=x_bins, density=True)
    hist_x_crystal, _ = np.histogram(df_crystal[x_col], bins=x_bins, density=True)
    hist_y_noncrystal, _ = np.histogram(df_noncrystal[y_col], bins=y_bins, density=True)
    hist_y_crystal, _ = np.histogram(df_crystal[y_col], bins=y_bins, density=True)
    
    # Grid layout settings
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(
        2, 5, 
        width_ratios=[1.0, 4, 4, 0.1, 0.15],
        height_ratios=[1.2, 4],
        wspace=0.1, hspace=0.05
    )
    
    # Common y-axis histogram (Left side)
    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.barh(y_centers, hist_y_noncrystal, height=dy, color='gray', alpha=0.5, align='center')
    ax_left.barh(y_centers, hist_y_crystal, height=dy, color='lightblue', alpha=0.8, align='center')
    ax_left.set_xlabel("Normalized frequency", fontsize=label_fontsize)
    ax_left.invert_xaxis()
    ax_left.set_ylim(y_bins[0], y_bins[-1])
    ax_left.tick_params(axis='both', labelsize=tick_fontsize)
    ax_left.yaxis.tick_right()
    ax_left.set_yticklabels([])
    
    # Adjust position
    pos = ax_left.get_position()
    ax_left.set_position([pos.x0 - 0.04, pos.y0, pos.width, pos.height])
    
    # Top x-axis histograms
    # For non-crystalline (Left)
    ax_top_noncrystal = fig.add_subplot(gs[0, 1])
    ax_top_noncrystal.bar(x_centers, hist_x_noncrystal, width=dx, color='gray', alpha=0.5)
    ax_top_noncrystal.set_xlim(0.8, 1.5)
    ax_top_noncrystal.tick_params(axis='both', labelsize=tick_fontsize)
    ax_top_noncrystal.tick_params(labelbottom=False)
    ax_top_noncrystal.set_ylabel("Normalized frequency", fontsize=label_fontsize)
    
    # For crystalline (Right)
    ax_top_crystal = fig.add_subplot(gs[0, 2])
    ax_top_crystal.bar(x_centers, hist_x_crystal, width=dx, color='lightblue', alpha=0.8)
    ax_top_crystal.set_xlim(0.8, 1.5)
    ax_top_crystal.tick_params(axis='both', labelsize=tick_fontsize)
    ax_top_crystal.tick_params(labelbottom=False)
    ax_top_crystal.yaxis.tick_right()
    ax_top_crystal.set_ylabel("Normalized frequency", fontsize=label_fontsize)
    ax_top_crystal.yaxis.set_label_position("right")
    
    # Main heatmaps
    # Heatmap for non-crystalline (Left)
    ax_main_noncrystal = fig.add_subplot(gs[1, 1])
    norm_noncrystal = mcolors.PowerNorm(
        gamma=0.5, 
        vmin=heat_noncrystal_norm.min(), 
        vmax=heat_noncrystal_norm.max()
    )
    im_left = ax_main_noncrystal.imshow(
        heat_noncrystal_norm,
        origin='lower',
        aspect='auto',
        extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
        cmap='inferno',
        norm=norm_noncrystal
    )
    ax_main_noncrystal.set_xlabel("Density", fontsize=label_fontsize)
    ax_main_noncrystal.set_ylabel("Predicted non-resistance probability", fontsize=label_fontsize)
    ax_main_noncrystal.set_xlim(0.8, 1.5)
    ax_main_noncrystal.set_ylim(y_bins[0], y_bins[-1])
    ax_main_noncrystal.tick_params(axis='both', labelsize=tick_fontsize)
    
    # Heatmap for crystalline (Right)
    ax_main_crystal = fig.add_subplot(gs[1, 2])
    norm_crystal = mcolors.PowerNorm(
        gamma=0.5, 
        vmin=heat_crystal_norm.min(), 
        vmax=heat_crystal_norm.max()
    )
    im_right = ax_main_crystal.imshow(
        heat_crystal_norm,
        origin='lower',
        aspect='auto',
        extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
        cmap='inferno',
        norm=norm_crystal
    )
    ax_main_crystal.set_xlabel("Density", fontsize=label_fontsize)
    ax_main_crystal.set_xlim(0.8, 1.5)
    ax_main_crystal.set_ylim(y_bins[0], y_bins[-1])
    ax_main_crystal.tick_params(axis='both', labelsize=tick_fontsize)
    ax_main_crystal.yaxis.tick_right()
    ax_main_crystal.yaxis.set_label_position("right")
    
    # Fix X-axis ticks
    ax_main_crystal.set_xticks(np.round(np.linspace(0.8, 1.5, 8), 2))
    ax_main_crystal.tick_params(axis='x', labelsize=tick_fontsize)
    
    # Colorbar
    ax_cbar = fig.add_subplot(gs[:, 4])
    cbar = plt.colorbar(im_left, cax=ax_cbar)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    cbar.set_label('Normalized frequency (PowerNorm γ=0.5)', fontsize=label_fontsize)
    ax_cbar.set_xticks([])
    ax_cbar.set_xticklabels([])
    
    # Layout adjustment
    plt.tight_layout()
    
    # Save if directory provided
    if save_dir:
        import os
        filename_jpg = os.path.join(save_dir, "fig6.jpg")
        filename_svg = os.path.join(save_dir, "fig6.svg")
        plt.savefig(filename_jpg, dpi=300, bbox_inches="tight")
        plt.savefig(filename_svg, dpi=300, bbox_inches="tight")
        print(f"Figure 6 saved to {filename_jpg} and {filename_svg}")
    
    return fig


def remove_ff_prefix(label: str) -> str:
    """
    Remove FF_ prefix and _solvent suffix from feature labels.
    
    Parameters
    ----------
    label : str
        Feature label to clean
    
    Returns
    -------
    str
        Cleaned label
    """
    if label.startswith("FF_"):
        label = label[len("FF_"):]
    if label.endswith("_solvent"):
        label = label[:-len("_solvent")]
    return label


def set_custom_yticks_fixed(ax: 'Axes') -> None:
    """
    Set custom y-axis ticks with lower limit fixed at -0.05 and steps of 0.1.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib Axes object
    """
    _, cur_ymax = ax.get_ylim()
    new_ymax = np.ceil(cur_ymax * 10) / 10
    ax.set_ylim(-0.05, new_ymax)
    yticks = np.arange(0, new_ymax + 0.05, 0.1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.1f}" for y in yticks])


def generate_figure8_model_comparison(
    df_vPolymer: pd.DataFrame,
    FF_solvent_cols: list,
    save_dir: str = None,
    base_font_size: int = 12,
    Th_high: float = 0.9,
    Th_low: float = 0.1
) -> Figure:
    """
    Generate Figure 8: Comprehensive Model Comparison.
    
    This function creates a 2-panel figure showing:
    (a) Average plot of polar force field parameters
    (b) Average plot of K_angle force field parameters
    
    Each panel shows mean ± 1σ for resistant vs non-resistant polymers.
    
    Parameters
    ----------
    df_vPolymer : pd.DataFrame
        DataFrame containing virtual polymer data with force field parameters
        and resistance predictions
    FF_solvent_cols : list
        List of force field solvent column names
    save_dir : str, optional
        Directory to save the figure
    base_font_size : int, optional
        Base font size for labels
    Th_high : float, optional
        High threshold for non-resistant classification (default: 0.9)
    Th_low : float, optional
        Low threshold for resistant classification (default: 0.1)
    
    Returns
    -------
    Figure
        Matplotlib Figure object
    """
    from matplotlib.patches import Patch
    
    # Font size settings
    title_font_size = base_font_size * 1.2
    xlabel_font_size = base_font_size * 1.2
    ylabel_font_size = base_font_size * 1.2
    tick_font_size = base_font_size
    tick_font_size_y = int(base_font_size * 0.9)
    legend_font_size = base_font_size * 0.9
    
    plt.rcParams.update({'font.size': base_font_size})
    
    # Extract necessary data
    df_class1_prob = df_vPolymer['resistance_pred']
    vpolymer_features = df_vPolymer[FF_solvent_cols]
    
    # Extract specific feature groups
    df_polar = vpolymer_features.filter(regex=f"^FF_polar_")
    df_kangle = vpolymer_features.filter(regex=f"^FF_k_angle_")
    
    # Extract conditions for each group
    df_polar_prob1 = df_polar[df_class1_prob >= Th_high]
    df_polar_prob0 = df_polar[df_class1_prob <= Th_low]
    df_kangle_prob1 = df_kangle[df_class1_prob >= Th_high]
    df_kangle_prob0 = df_kangle[df_class1_prob <= Th_low]
    
    n1_polar, n0_polar = len(df_polar_prob1), len(df_polar_prob0)
    n1_kangle, n0_kangle = len(df_kangle_prob1), len(df_kangle_prob0)
    
    # Labels for x-axis
    labels_polar = [remove_ff_prefix(col) for col in df_polar.columns]
    labels_kangle = [remove_ff_prefix(col) for col in df_kangle.columns]
    
    # Label functions
    def label_R(n):
        return f"Resistant (p<{Th_low}, n={n:,})"
    
    def label_NR(n):
        return f"Non-resistant (p>{Th_high}, n={n:,})"
    
    sigma_patch = Patch(facecolor='0.5', alpha=0.2, edgecolor='none')
    
    # Create subplots: 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # ==== (a) Plot of polar ====
    ax = axes[0]
    x_polar = np.arange(len(df_polar.columns))
    df_polar_prob1_mean = df_polar_prob1.mean(axis=0)
    df_polar_prob1_std = df_polar_prob1.std(axis=0)
    df_polar_prob0_mean = df_polar_prob0.mean(axis=0)
    df_polar_prob0_std = df_polar_prob0.std(axis=0)
    
    # Resistant: Line and ±1σ
    line_R_polar, = ax.plot(
        x_polar, df_polar_prob0_mean, color='green', lw=2,
        label=label_R(n0_polar)
    )
    ax.fill_between(
        x_polar,
        df_polar_prob0_mean - df_polar_prob0_std,
        df_polar_prob0_mean + df_polar_prob0_std,
        color='green', alpha=0.2, label="_nolegend_"
    )
    
    # Non-resistant
    line_NR_polar, = ax.plot(
        x_polar, df_polar_prob1_mean, color='red', lw=2,
        label=label_NR(n1_polar)
    )
    ax.fill_between(
        x_polar,
        df_polar_prob1_mean - df_polar_prob1_std,
        df_polar_prob1_mean + df_polar_prob1_std,
        color='red', alpha=0.2, label="_nolegend_"
    )
    
    ax.set_xticks(x_polar)
    ax.set_xticklabels(labels_polar, rotation=90, fontsize=tick_font_size)
    plt.sca(ax)
    set_custom_yticks_fixed(ax)
    
    ax.set_ylabel("Force field kernel mean descriptor", fontsize=ylabel_font_size)
    ax.set_title("Average plot of polar", fontsize=title_font_size)
    ax.legend([line_R_polar, line_NR_polar, sigma_patch],
              [label_R(n0_polar), label_NR(n1_polar), "±1σ"],
              loc='upper right', frameon=False)
    
    # ==== (b) Plot of K_angle ====
    ax = axes[1]
    x_kangle = np.arange(len(df_kangle.columns))
    df_kangle_prob1_mean = df_kangle_prob1.mean(axis=0)
    df_kangle_prob1_std = df_kangle_prob1.std(axis=0)
    df_kangle_prob0_mean = df_kangle_prob0.mean(axis=0)
    df_kangle_prob0_std = df_kangle_prob0.std(axis=0)
    
    # Resistant
    line_R_k, = ax.plot(
        x_kangle, df_kangle_prob0_mean, color='green', lw=2,
        label=label_R(n0_kangle)
    )
    ax.fill_between(
        x_kangle,
        df_kangle_prob0_mean - df_kangle_prob0_std,
        df_kangle_prob0_mean + df_kangle_prob0_std,
        color='green', alpha=0.2, label="_nolegend_"
    )
    
    # Non-resistant
    line_NR_k, = ax.plot(
        x_kangle, df_kangle_prob1_mean, color='red', lw=2,
        label=label_NR(n1_kangle)
    )
    ax.fill_between(
        x_kangle,
        df_kangle_prob1_mean - df_kangle_prob1_std,
        df_kangle_prob1_mean + df_kangle_prob1_std,
        color='red', alpha=0.2, label="_nolegend_"
    )
    
    ax.set_xticks(x_kangle)
    ax.set_xticklabels(labels_kangle, rotation=90, fontsize=tick_font_size)
    plt.sca(ax)
    set_custom_yticks_fixed(ax)
    
    ax.set_title(r"Average plot of $K_{angle}$", fontsize=title_font_size)
    ax.legend([line_R_k, line_NR_k, sigma_patch],
              [label_R(n0_kangle), label_NR(n1_kangle), "±1σ"],
              loc='upper right', frameon=False)
    
    # Add labels (a) and (b) to the top left of each subplot
    pos0 = axes[0].get_position()
    pos1 = axes[1].get_position()
    
    fig.text(
        pos0.x0 - 0.115, pos0.y1 + 0.09, "(a)",
        transform=fig.transFigure,
        fontsize=title_font_size,
        fontweight="normal",
        fontfamily="Arial",
        ha="left", va="top"
    )
    fig.text(
        pos1.x0 - 0.045, pos1.y1 + 0.09, "(b)",
        transform=fig.transFigure,
        fontsize=title_font_size,
        fontweight="normal",
        fontfamily="Arial",
        ha="left", va="top"
    )
    
    plt.tight_layout()
    
    # Save if directory provided
    if save_dir:
        import os
        filename_jpg = os.path.join(save_dir, "fig8.jpg")
        filename_svg = os.path.join(save_dir, "fig8.svg")
        filename_pdf = os.path.join(save_dir, "fig8.pdf")
        plt.savefig(filename_jpg, dpi=300, bbox_inches="tight")
        plt.savefig(filename_svg, dpi=300, bbox_inches="tight")
        plt.savefig(filename_pdf, dpi=300, bbox_inches="tight")
        print(f"Figure 8 saved to {filename_jpg}, {filename_svg}, and {filename_pdf}")
    
    return fig


def generate_figure7a_heatmap(
    df_heatmap: pd.DataFrame,
    FF_solvent_cols: list,
    save_dir: str = None,
    base_font_size: int = 28
) -> Figure:
    """
    Generate Figure 7a: Chi Parameter Prediction Model Performance (Heatmap).
    
    This function creates a detailed heatmap of force field descriptors,
    and optionally saves x-axis labels as a separate SVG.
    
    Parameters
    ----------
    df_heatmap : pd.DataFrame
        DataFrame containing force field columns
    FF_solvent_cols : list
        List of force field solvent column names
    save_dir : str, optional
        Directory to save the figure
    base_font_size : int, optional
        Base font size for labels
    
    Returns
    -------
    Figure
        Matplotlib Figure object
    """
    import matplotlib as mpl
    
    mpl.rcParams["svg.fonttype"] = "none"
    
    title_font_size = base_font_size * 1.4
    xlabel_font_size = base_font_size * 1.0
    ylabel_font_size = base_font_size * 1.2
    tick_font_size = base_font_size * 0.4
    colorbar_label_font_size = base_font_size * 1.1
    
    # Create heatmap data
    heatmap_data = df_heatmap[FF_solvent_cols].reset_index(drop=True)
    
    # Clean column names
    new_cols = []
    for col in heatmap_data.columns:
        new_col = col
        if new_col.startswith("FF_"):
            new_col = new_col[len("FF_"):]
        if new_col.endswith("_solvent"):
            new_col = new_col[:-len("_solvent")]
        new_cols.append(new_col)
    heatmap_data.columns = new_cols
    
    # Draw heatmap
    fig = plt.figure(figsize=(32, 8))
    ax = sns.heatmap(heatmap_data, cmap='magma')
    plt.title('Heatmap of kernel mean descriptors', fontsize=title_font_size)
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_xticks(np.arange(len(heatmap_data.columns)) + 0.5)
    xtick_labels = [label.replace("_", " ") for label in heatmap_data.columns]
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=tick_font_size)
    
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=colorbar_label_font_size)
    plt.tight_layout()
    
    # Save figures
    if save_dir:
        import os
        filename_jpg = os.path.join(save_dir, "fig7a_FF_solvent_heatmap.jpg")
        plt.savefig(filename_jpg, dpi=300, bbox_inches="tight")
        
        filename_svg = os.path.join(save_dir, "fig7a_FF_solvent_heatmap.svg")
        plt.savefig(filename_svg, dpi=300, bbox_inches="tight")
        
        # Save x-axis labels separately
        fig_x, ax_x = plt.subplots(figsize=(32, 1))
        ax_x.axis("off")
        for i, label in enumerate(xtick_labels):
            x = i / (len(xtick_labels) - 1) if len(xtick_labels) > 1 else 0.5
            ax_x.text(x, 0.5, label, ha="center", va="top",
                      fontsize=tick_font_size, rotation=90, transform=ax_x.transAxes)
        filename_only_xaxis = os.path.join(save_dir, "fig7a_only_xaxis_labels.svg")
        plt.savefig(filename_only_xaxis, bbox_inches="tight")
        
        print(f"Figure 7a saved to {filename_svg}")
    
    return fig


def generate_figure7b_resistance_analysis(
    df_PE_vSolv: pd.DataFrame,
    FF_solvent_cols: list,
    save_dir: str = None,
    base_font_size: int = 28
) -> tuple:
    """
    Generate Figure 7b: Chi Parameter Relationship Analysis.
    
    This function creates two plots:
    1. Resistance prediction probability vs. Index plot
    2. Heatmap of force field descriptors
    
    Parameters
    ----------
    df_PE_vSolv : pd.DataFrame
        DataFrame containing PE-solvent data with resistance_pred column
    FF_solvent_cols : list
        List of force field solvent column names
    save_dir : str, optional
        Directory to save the figures
    base_font_size : int, optional
        Base font size for labels
    
    Returns
    -------
    tuple
        (fig_plot, fig_heatmap) Matplotlib Figure objects
    """
    import matplotlib.ticker as mtick
    
    plt.rcParams.update({'font.size': base_font_size})
    
    # Font size settings
    title_font_size = base_font_size * 1.2
    xlabel_font_size = base_font_size * 0.9
    ylabel_font_size = base_font_size * 0.9
    xticks_font_size = base_font_size * 0.7
    yticks_font_size = base_font_size * 0.7
    
    # ===== Part 1: Resistance prediction probability vs Index =====
    df_heatmap = df_PE_vSolv.copy()
    df_heatmap.sort_values("resistance_pred", inplace=True)
    
    fig_plot = plt.figure(figsize=(12, 8))
    plt.plot(
        df_heatmap['resistance_pred'],
        df_heatmap.reset_index(drop=True).index,
        marker='o',
        linestyle='-'
    )
    
    plt.title('Resistance prediction probability', fontsize=title_font_size)
    plt.xlabel('Predicted non-resistance probabilitiy', fontsize=xlabel_font_size)
    plt.ylabel('Index', fontsize=ylabel_font_size)
    plt.xticks(fontsize=xticks_font_size)
    plt.yticks(fontsize=yticks_font_size)
    
    ax = plt.gca()
    ax.invert_yaxis()
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f'{int(x):,}'))
    plt.tight_layout()
    
    if save_dir:
        import os
        filename_svg = os.path.join(save_dir, "fig7b_FF_solvent_resistance_pred.svg")
        filename_jpg = os.path.join(save_dir, "fig7b_FF_solvent_resistance_pred.jpg")
        plt.savefig(filename_svg, dpi=300, bbox_inches="tight")
        plt.savefig(filename_jpg, dpi=300, bbox_inches="tight")
        print(f"Figure 7b (plot) saved to {filename_svg} and {filename_jpg}")
    
    # ===== Part 2: Heatmap (commented out in original, but included here) =====
    tick_font_size = base_font_size * 0.4
    colorbar_label_font_size = base_font_size * 1.1
    
    heatmap_data = df_heatmap[FF_solvent_cols].reset_index(drop=True)
    
    # Clean column names
    new_cols = []
    for col in heatmap_data.columns:
        new_col = col
        if new_col.startswith("FF_"):
            new_col = new_col[len("FF_"):]
        if new_col.endswith("_solvent"):
            new_col = new_col[:-len("_solvent")]
        new_cols.append(new_col)
    heatmap_data.columns = new_cols
    
    fig_heatmap = plt.figure(figsize=(32, 8))
    ax = sns.heatmap(heatmap_data, cmap='magma')
    plt.title('Heatmap of kernel mean descriptors', fontsize=title_font_size)
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_xticks(np.arange(len(heatmap_data.columns)) + 0.5)
    xtick_labels = [label.replace("_", " ") for label in heatmap_data.columns]
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=tick_font_size)
    
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=colorbar_label_font_size)
    plt.tight_layout()
    
    # Note: Heatmap saving is commented out in the original Cell 230
    # Uncomment if needed:
    # if save_dir:
    #     filename = os.path.join(save_dir, "fig7a_FF_solvent_heatmap.svg")
    #     plt.savefig(filename, dpi=300, bbox_inches="tight")
    
    return fig_plot, fig_heatmap


def generate_figure7c_chi_mic(
    df_vsolvent_MIC: pd.DataFrame,
    FF_solvent_cols: list,
    save_dir: str = None,
    base_font_size: int = 18
) -> Figure:
    """
    Generate Figure 7c: Chi Parameter Distribution by Force Field Parameter Type.
    
    This function creates a bar plot showing MIC values for different force field
    parameter groups (mass, charge, epsilon, sigma, K_bond, r0, polar, K_angle,
    theta0, K_dihedral).
    
    Parameters
    ----------
    df_vsolvent_MIC : pd.DataFrame
        DataFrame containing MIC scores for features
    FF_solvent_cols : list
        List of force field solvent column names
    save_dir : str, optional
        Directory to save the figure
    base_font_size : int, optional
        Base font size for labels
    
    Returns
    -------
    Figure
        Matplotlib Figure object
    """
    import matplotlib.ticker as ticker
    
    plt.rcParams.update({'font.size': base_font_size, 'font.family': 'Arial'})
    
    # Data preparation
    mic_df = df_vsolvent_MIC.copy()
    mic_subset = mic_df[mic_df["feature"].isin(FF_solvent_cols)].copy()
    mic_subset["feature"] = pd.Categorical(
        mic_subset["feature"], categories=FF_solvent_cols, ordered=True
    )
    mic_subset = mic_subset.sort_values("feature")
    
    features = mic_subset["feature"]
    mic = mic_subset["MIC"]
    y_pos = np.arange(len(features))
    colors = []
    
    # Color and label settings for each group
    group_centers = {}
    group_labels = {}
    current_group = None
    group_start = 0
    
    for i, feature in enumerate(features):
        if feature.startswith("FF_mass_"):
            color, group, label = "royalblue", "FF_mass", r"$\mathrm{mass}$"
        elif feature.startswith("FF_charge_"):
            color, group, label = "orangered", "FF_charge", r"$\mathrm{charge}$"
        elif feature.startswith("FF_epsilon_"):
            color, group, label = "green", "FF_epsilon", r"$\epsilon$"
        elif feature.startswith("FF_sigma_"):
            color, group, label = "red", "FF_sigma", r"$\sigma$"
        elif feature.startswith("FF_k_bond_"):
            color, group, label = "purple", "FF_k_bond", r"$K_{bond}$"
        elif feature.startswith("FF_r0_"):
            color, group, label = "brown", "FF_r0", r"$r_0$"
        elif feature.startswith("FF_polar_"):
            color, group, label = "deeppink", "FF_polar", r"$\mathrm{polar}$"
        elif feature.startswith("FF_k_angle_"):
            color, group, label = "dimgray", "FF_k_angle", r"$K_{angle}$"
        elif feature.startswith("FF_theta0_"):
            color, group, label = "olive", "FF_theta0", r"$\theta_0$"
        elif feature.startswith("FF_k_dih_"):
            color, group, label = "darkcyan", "FF_k_dih", r"$K_{dihedral}$"
        else:
            color, group, label = "gray", "Other", r"$Other$"
        colors.append(color)
        
        if group != current_group:
            if current_group is not None:
                center = (group_start + i - 1) / 2
                group_centers[current_group] = center
            current_group = group
            group_start = i
            group_labels[group] = label
    
    # Add center of the last group
    if current_group is not None:
        center = (group_start + len(features) - 1) / 2
        group_centers[current_group] = center
    
    # Create figure
    fig = plt.figure(figsize=(14, 6))
    plt.bar(y_pos, mic, align='center', alpha=0.7, color=colors)
    
    # Axis labels and title
    plt.xticks(
        ticks=list(group_centers.values()),
        labels=[group_labels[g] for g in group_centers.keys()],
        fontsize=base_font_size * 0.9
    )
    plt.yticks(fontsize=base_font_size)
    plt.ylabel('MIC', fontsize=base_font_size * 1.2)
    plt.title('MIC of resistance prediction probability', fontsize=base_font_size * 1.4)
    
    plt.tight_layout()
    plt.xlim(-0.5, len(features) - 0.5)
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.gca().tick_params(axis='y', which='both', direction='out', length=8, width=1)
    
    # Save if directory provided
    if save_dir:
        import os
        filename_svg = os.path.join(save_dir, "fig7c_FF_solvent_MIC.svg")
        filename_jpg = os.path.join(save_dir, "fig7c_FF_solvent_MIC.jpg")
        plt.savefig(filename_svg, dpi=300, bbox_inches="tight")
        plt.savefig(filename_jpg, dpi=300, bbox_inches="tight")
        print(f"Figure 7c saved to {filename_svg} and {filename_jpg}")
    
    return fig


def generate_figure7d_pca_scatter(
    df_PE_vSolv: pd.DataFrame,
    features: list,
    save_dir: str = None,
    base_font_size: int = 24
) -> Figure:
    """
    Generate Figure 7d: Chi Parameter Feature Importance (PCA Scatter Plot).
    
    This function performs PCA on the features and creates a scatter plot
    colored by resistance prediction probability.
    
    Parameters
    ----------
    df_PE_vSolv : pd.DataFrame
        DataFrame containing PE-solvent data with features and resistance_pred
    features : list
        List of feature column names to use for PCA
    save_dir : str, optional
        Directory to save the figure
    base_font_size : int, optional
        Base font size for labels
    
    Returns
    -------
    Figure
        Matplotlib Figure object
    """
    from sklearn.decomposition import PCA
    from matplotlib.colors import LinearSegmentedColormap
    
    plt.rcParams.update({'font.size': base_font_size})
    
    title_font_size = base_font_size * 1.2
    xlabel_font_size = base_font_size * 1.0
    ylabel_font_size = base_font_size * 1.0
    tick_font_size = base_font_size * 0.8
    colorbar_label_font_size = base_font_size * 0.8
    
    # Perform PCA
    X = df_PE_vSolv[features].values
    pca = PCA(n_components=2)
    PC = pca.fit_transform(X)
    
    # Add PCA results to DataFrame (matching original behavior)
    df_PE_vSolv["PC1"] = PC[:, 0]
    df_PE_vSolv["PC2"] = PC[:, 1]
    
    # Explained variance ratio
    pc1_explained = pca.explained_variance_ratio_[0] * 100
    pc2_explained = pca.explained_variance_ratio_[1] * 100
    
    # Define custom colormap
    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_lightgreen_lightcoral', ['green', 'lightcoral']
    )
    
    # Create scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    sc = ax.scatter(
        df_PE_vSolv["PC1"],
        df_PE_vSolv["PC2"],
        c=df_PE_vSolv["resistance_pred"],
        cmap=custom_cmap,
        s=30,
        alpha=0.7,
        vmin=0,
        vmax=1
    )
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Resistance prediction probability", fontsize=colorbar_label_font_size)
    cbar.ax.tick_params(labelsize=tick_font_size)
    
    # Set axis ranges and ticks
    ax.set_xlim(-1, 2.5)
    ax.set_ylim(-1, 2)
    x_ticks = np.linspace(-1, 2.5, 6)
    y_ticks = np.linspace(-1, 2, 6)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    
    # Labels and title
    ax.set_xlabel(f"PC1 ({pc1_explained:.1f}%)", fontsize=xlabel_font_size)
    ax.set_ylabel(f"PC2 ({pc2_explained:.1f}%)", fontsize=ylabel_font_size)
    ax.set_title("Scatter plot in PCA space", fontsize=title_font_size)
    
    # Enhance frame
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save if directory provided
    if save_dir:
        import os
        filename_svg = os.path.join(save_dir, "fig7d_PC_heatmap.svg")
        filename_jpg = os.path.join(save_dir, "fig7d_PC_heatmap.jpg")
        plt.savefig(filename_svg, dpi=300, bbox_inches="tight")
        plt.savefig(filename_jpg, dpi=300, bbox_inches="tight")
        print(f"Figure 7d saved to {filename_svg} and {filename_jpg}")
    
    return fig


def generate_figure9_chi_analysis(
    valid_data_cluster: pd.DataFrame,
    valid_info_cluster: pd.DataFrame,
    gbdt_model,
    all_features_combinations: list,
    feature_No: int,
    df_dict: dict,
    symbol_order: list,
    save_dir: str = None,
    base_fontsize: int = 20
) -> Figure:
    """
    Generate Figure 9: Crystallinity Parameter Analysis (χ distribution).
    
    This function creates a two-panel figure:
    (a) Left: χ histogram for entire chemical resistance dataset
    (b) Right: χ histogram for PE polymer specifically
    
    Parameters
    ----------
    valid_data_cluster : pd.DataFrame
        DataFrame with features for all data
    valid_info_cluster : pd.DataFrame
        DataFrame with cluster and label information
    gbdt_model : model
        Trained GBDT model with predict_proba method
    all_features_combinations : list
        List of feature combinations
    feature_No : int
        Index of feature combination to use
    df_dict : dict
        Dictionary mapping polymer symbols to their DataFrames
    symbol_order : list
        Order of symbols for processing
    save_dir : str, optional
        Directory to save the figure
    base_fontsize : int, optional
        Base font size for labels
    
    Returns
    -------
    Figure
        Matplotlib Figure object
    """
    import matplotlib.ticker as mticker
    
    def plot_chi_hist_from_features_ax(
        ax, df_features, df_info, model, all_features,
        exclude_cols=['n_atom_radonpy_polymer', 'mol_weight_radonpy_polymer'],
        chi_col="chi", binary_col="resistance_binary",
        xlim=(-0.5, 7), base_fontsize=20
    ):
        """Helper function to plot χ histogram from features."""
        features = [col for col in all_features if col not in exclude_cols]
        train_data = df_features.copy()[features]
        X_train_exp = train_data[features]
        train_data["resistance_pred"] = model.predict_proba(X_train_exp)[:, 1]
        label = df_info[binary_col]
        chi = X_train_exp[chi_col]
        chi_0 = chi[label == 0]
        chi_1 = chi[label == 1]
        n_0, n_1 = len(chi_0), len(chi_1)
        n_all = n_0 + n_1
        bins = np.linspace(xlim[0], xlim[1], 51)
        
        ax.hist(chi_0, bins=bins, color="green", alpha=0.3,
                label=f"Resistant (n = {n_0:,})", density=True)
        ax.hist(chi_1, bins=bins, color="red", alpha=0.3,
                label=f"Non-resistant (n = {n_1:,})", density=True)
        
        ax.set_title(f"Histogram of $\\mathit{{\\chi}}$ for {n_all:,} data in chemical resistance dataset",
                     fontsize=base_fontsize)
        ax.set_xlabel(r"$\mathit{\chi}$", fontsize=base_fontsize)
        ax.set_ylabel("Normalized frequency", fontsize=base_fontsize)
        ax.tick_params(axis='both', labelsize=base_fontsize)
        ax.legend(fontsize=base_fontsize)
        ax.set_xlim(xlim)
        ax.figure.tight_layout()
    
    def plot_chi_hist_by_binary_ax(
        ax, df, symbol, chi_col="chi", binary_col="resistance_pred_binary",
        xlim=None, base_fontsize=20
    ):
        """Helper function to plot χ histogram by binary classification."""
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
        
        ax.hist(chi_0, bins=bins, alpha=0.3, label=f"Resistant (n = {n_0:,})",
                density=True, color="green")
        ax.hist(chi_1, bins=bins, alpha=0.3, label=f"Non-resistant (n = {n_1:,})",
                density=True, color="red")
        
        ax.set_title(f"Histogram of $\\mathit{{\\chi}}$ for {symbol} vs {n_all:,} solvent",
                     fontsize=base_fontsize)
        ax.set_xlabel(r"$\mathit{\chi}$", fontsize=base_fontsize)
        ax.set_ylabel("Normalized frequency", fontsize=base_fontsize)
        ax.tick_params(axis='both', labelsize=base_fontsize)
        
        # x-axis tick setting
        x_min, x_max = ax.get_xlim()
        raw_step = (x_max - x_min) / 5
        temp_step = np.round(raw_step, 1)
        if temp_step == 0:
            temp_step = 0.1
        step = np.ceil(temp_step / 0.5) * 0.5
        
        if x_min >= 0:
            new_x_min = 0.0
        else:
            new_x_min = np.floor(x_min / step) * step
        
        if x_max <= 0:
            new_x_max = 0.0
        else:
            new_x_max = np.ceil(x_max / step) * step
        
        xticks = np.arange(new_x_min, new_x_max + step / 10, step)
        if not any(np.isclose(xticks, 0.0, atol=1e-8)):
            xticks = np.sort(np.append(xticks, 0.0))
        ax.set_xticks(xticks)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
        
        # y-axis tick setting
        y_min, y_max = ax.get_ylim()
        step_y = np.round((y_max - y_min) / 4, 1)
        if step_y == 0:
            step_y = 0.1
        if y_min >= 0:
            new_y_min = 0.0
        else:
            new_y_min = np.floor(y_min / step_y) * step_y
        if y_max <= 0:
            new_y_max = 0.0
        else:
            new_y_max = np.ceil(y_max / step_y) * step_y
        yticks = np.arange(new_y_min, new_y_max + step_y / 10, step_y)
        if not any(np.isclose(yticks, 0.0, atol=1e-8)):
            yticks = np.sort(np.append(yticks, 0.0))
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        
        ax.legend(fontsize=base_fontsize)
        ax.set_xlim(xlim)
        ax.figure.tight_layout()
    
    def get_final_symbol(symbol, symbol_order):
        """Helper function to get final symbol name."""
        for s in symbol_order:
            if symbol in s:
                return s
        return symbol
    
    # Create figure with two subplots
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left graph: entire dataset
    features_list = list(all_features_combinations[feature_No])
    plot_chi_hist_from_features_ax(
        ax=ax_left,
        df_features=valid_data_cluster,
        df_info=valid_info_cluster,
        model=gbdt_model,
        all_features=features_list,
        exclude_cols=['n_atom_radonpy_polymer', 'mol_weight_radonpy_polymer'],
        chi_col="chi",
        binary_col="resistance_binary",
        xlim=(-3, 7),
        base_fontsize=base_fontsize
    )
    
    # Right graph: PE specific
    if "PE" in df_dict:
        final_symbol = get_final_symbol("PE", symbol_order)
        display_symbol = final_symbol.replace("solv", "solvent", 1) if final_symbol.startswith("solv") else final_symbol
        
        plot_chi_hist_by_binary_ax(
            ax=ax_right,
            df=df_dict["PE"],
            symbol="PE",
            chi_col="chi",
            binary_col="resistance_pred_binary",
            xlim=None,
            base_fontsize=base_fontsize
        )
    else:
        ax_right.text(0.5, 0.5, "Data corresponding to symbol 'PE'\nwas not found",
                      horizontalalignment='center', verticalalignment='center',
                      fontsize=base_fontsize)
        ax_right.set_axis_off()
    
    # Add (a) and (b) labels
    pos_left = ax_left.get_position()
    pos_right = ax_right.get_position()
    
    fig.text(
        pos_left.x0 - 0.045, pos_left.y1 + 0.01, "(a)",
        transform=fig.transFigure,
        fontsize=base_fontsize + 4,
        fontweight="normal",
        fontfamily="Arial",
        ha="left", va="bottom"
    )
    fig.text(
        pos_right.x0 - 0.045, pos_right.y1 + 0.01, "(b)",
        transform=fig.transFigure,
        fontsize=base_fontsize + 4,
        fontweight="normal",
        fontfamily="Arial",
        ha="left", va="bottom"
    )
    
    # Save if directory provided
    if save_dir:
        import os
        for ext in ['jpg', 'svg', 'pdf']:
            filename = os.path.join(save_dir, f"fig9.{ext}")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Figure 9 saved to {save_dir}")
    
    return fig


def generate_supplementary_s2_cluster_heatmaps(
    valid_info_cluster: pd.DataFrame,
    df_sorted: pd.DataFrame,
    solvent_name_dict: dict,
    save_dir: str = None
) -> None:
    """
    Generate Supplementary Figure S2: Cluster Heatmaps.
    
    Creates heatmaps for each solvent cluster showing resistance patterns
    across polymers. Generates multiple figures with 2 clusters per figure.
    
    Parameters
    ----------
    valid_info_cluster : pd.DataFrame
        DataFrame with clustered validation information and resistance data
    df_sorted : pd.DataFrame
        Sorted DataFrame defining the order of polymers
    solvent_name_dict : dict
        Dictionary mapping solvent SMILES to names with Japanese abbreviations
    save_dir : str, optional
        Directory to save the figures
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import ListedColormap
    import os
    
    # Increase overall font size
    plt.rcParams.update({'font.size': 16})
    
    # Preparation
    os.makedirs(save_dir, exist_ok=True)
    
    # Define polymer order from polymer_rank
    symbol_order = df_sorted.sort_values('polymer_rank')['symbol'].unique()
    
    # List for changing color of specific horizontal axis labels (non-crystalline polymers)
    special_labels = ['PVA', 'PIB', 'PMMA', 'PVdC', 'PC', 'PVC', 'PS', 'PSF', 'PMP']
    
    # List of all clusters
    unique_clusters = sorted(valid_info_cluster["cluster_labels"].unique())
    n_clusters = len(unique_clusters)
    
    # Split clusters into groups of 2
    grouped_clusters = [unique_clusters[i:i+2] for i in range(0, n_clusters, 2)]
    
    # Counter for serial number
    file_no = 1
    
    # Draw as 2 rows 1 column subplots per group
    for group in grouped_clusters:
        # Skip if group has only 1 cluster (odd number of clusters)
        if len(group) == 1:
            print(f"Skipping cluster {group[0]} (single cluster in group)")
            continue
        
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 16))
        
        # Process for each cluster in the group
        for ax, cluster in zip(axs, group):
            # Extract data for the corresponding cluster
            cluster_data = valid_info_cluster[valid_info_cluster["cluster_labels"] == cluster].copy()
            cluster_data.sort_values(['smiles_solvent'], inplace=True)
            
            # Create pivot table
            pivot_data = cluster_data.pivot(index='smiles_solvent', columns='symbol', values='resistance_binary')
            
            # Match SMILES order to appearance order in DataFrame
            smiles_order = cluster_data['smiles_solvent'].unique()
            pivot_data = pivot_data.reindex(index=smiles_order)
            # Reorder columns by polymer_rank
            pivot_data = pivot_data.reindex(columns=symbol_order)
            
            # Draw heatmap
            cmap = ListedColormap([(0, 0.5, 0, 0.5), (1, 0.7, 0.7, 0.5)])
            sns.heatmap(
                pivot_data,
                ax=ax,
                cmap=cmap,
                cbar=False,
                annot=False,
                linewidths=0.1,
                linecolor='gray',
                vmin=0,
                vmax=1
            )
            
            # Set title and axis labels
            ax.set_title(f'Cluster {cluster}, Data count: {len(cluster_data)}', fontsize=20)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            
            # Change color of x-axis tick labels for non-crystalline polymers
            for label in ax.get_xticklabels():
                if label.get_text() in special_labels:
                    label.set_color('red')
            
            # Add chemical structure images to the left of y-axis labels (solvents)
            solvent_smiles_keys = list(pivot_data.index)
            add_structure_images(
                ax,
                solvent_smiles_keys,
                solvent_name_dict,
                get_rdkit_image,
                image_zoom=0.3,
                label_x=-0.13,
                image_x_offset=0.05,
                image_y_offset=0.0
            )
        
        fig.tight_layout(pad=2.0)
        
        # Save with cluster numbers and serial number in filename
        # Using S2_ prefix to distinguish from old resistance_matrix_cluster_* files
        group_str = "_".join(str(cl) for cl in group)
        svg_path = os.path.join(save_dir, f"S2_resistance_matrix_clusters_{group_str}_file{file_no}.svg")
        jpg_path = os.path.join(save_dir, f"S2_resistance_matrix_clusters_{group_str}_file{file_no}.jpg")
        
        save_as_png(fig, jpg_path)
        save_as_png(fig, svg_path)
        
        file_no += 1
    
    print(f"Generated {file_no-1} cluster heatmap figures for S2")


def get_final_symbol(cluster_name: str, symbol_order: list) -> str:
    """
    Format cluster name for output filename.
    
    Parameters
    ----------
    cluster_name : str
        String with extension removed (e.g., "cv_ABC" or "cv_solvXXX")
    symbol_order : list
        List of symbol order
    
    Returns
    -------
    str
        Formatted symbol name with number prefix if applicable
    """
    # Remove leading "cv_"
    if cluster_name.startswith("cv_"):
        raw_symbol = cluster_name[3:]
    else:
        raw_symbol = cluster_name
    
    # If cluster_name starts with "cv_solv", return as is
    if cluster_name.startswith("cv_solv"):
        return raw_symbol
    else:
        # If raw_symbol is in symbol_order list, assign its number
        symbol_list = list(symbol_order) if hasattr(symbol_order, '__array__') else symbol_order
        if symbol_list and raw_symbol in symbol_list:
            idx = symbol_list.index(raw_symbol)
            return f"{idx+1:02d}_{raw_symbol}"
        else:
            return raw_symbol


def generate_supplementary_s3_roc_confusion(
    plot_data_dir: str,
    output_dir: str,
    symbol_order: list = None,
    font_size: int = 16
) -> None:
    """
    Generate Supplementary Figure S3: ROC Curves and Confusion Matrices.
    
    For each cluster (polymer), creates a combined plot with:
    - Left: ROC curve (Train and Validation)
    - Right: Validation confusion matrix
    
    Parameters
    ----------
    plot_data_dir : str
        Directory containing plot data files (*_train_roc_data.json, *_val_roc_data.json, *_val_confusion_data.csv)
    output_dir : str
        Directory to save the output figures (will create S3 subfolder)
    symbol_order : list, optional
        Order of polymer symbols for numbering
    font_size : int, optional
        Font size for plots (default: 16)
    """
    import json
    import re
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import os
    
    # Create S3 subfolder in output directory
    s3_output_dir = os.path.join(output_dir, "S3")
    os.makedirs(s3_output_dir, exist_ok=True)
    
    # Function to combine ROC curve and confusion matrix into one image
    def plot_combined_cluster(cluster_name):
        """
        Create combined plot for one cluster.
        
        cluster_name: Cluster name (e.g., "cv_ABC" or "cv_solvXXX")
        """
        # Format symbol name
        final_symbol = get_final_symbol(cluster_name, symbol_order)
        display_symbol = final_symbol.replace("solv", "solvent", 1) if final_symbol.startswith("solv") else final_symbol
        clean_display_symbol = re.sub(r'^\d{2}_', '', display_symbol)
        
        # Create file paths
        train_json_file = os.path.join(plot_data_dir, f"{cluster_name}_train_roc_data.json")
        val_json_file   = os.path.join(plot_data_dir, f"{cluster_name}_val_roc_data.json")
        conf_csv_file   = os.path.join(plot_data_dir, f"{cluster_name}_val_confusion_data.csv")
        
        # File existence check
        if not (os.path.exists(train_json_file) and os.path.exists(val_json_file) and os.path.exists(conf_csv_file)):
            print(f"Required files not found for: {cluster_name}")
            return
        
        # Read ROC Curve data
        with open(train_json_file, 'r') as f:
            train_data = json.load(f)
        with open(val_json_file, 'r') as f:
            val_data = json.load(f)
        
        train_fpr = np.array(train_data['fpr'])
        train_tpr = np.array(train_data['tpr'])
        train_auc = train_data.get('auc', 0)
        
        val_fpr = np.array(val_data['fpr'])
        val_tpr = np.array(val_data['tpr'])
        val_auc = val_data.get('auc', 0)
        
        # Read Confusion Matrix (Validation)
        df_conf = pd.read_csv(conf_csv_file)
        cm = df_conf.values
        total_correct = np.trace(cm)
        total = np.sum(cm)
        micro_f1 = total_correct / total if total > 0 else 0
        
        # Create Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: ROC Curve
        ax_roc = axes[0]
        ax_roc.plot(train_fpr, train_tpr, color='blue', lw=2, 
                    label=f"Train ROC (AUC = {train_auc:.3f})")
        ax_roc.plot(val_fpr, val_tpr, color='orange', lw=2, 
                    label=f"Validation ROC (AUC = {val_auc:.3f})")
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
        ax_roc.set_xlabel("False Positive Rate", fontsize=font_size)
        ax_roc.set_ylabel("True Positive Rate", fontsize=font_size)
        ax_roc.set_title(f"ROC Curve - {clean_display_symbol}", fontsize=font_size+2)
        ax_roc.legend(loc="lower right", fontsize=font_size)
        ax_roc.tick_params(labelsize=font_size)
        
        # Right: Validation Confusion Matrix
        ax_conf = axes[1]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": font_size}, ax=ax_conf)
        title_conf = f"Validation Confusion Matrix - {clean_display_symbol}\n(micro F1: {micro_f1:.3f})"
        ax_conf.set_title(title_conf, fontsize=font_size+2)
        
        # Change axis labels
        ax_conf.set_xticklabels(["Predicted resistant (0)", "Predicted non-resistant (1)"], 
                                fontsize=font_size)
        ax_conf.set_yticklabels(["True resistant (0)", "True non-resistant (1)"], 
                                fontsize=font_size)
        ax_conf.set_xlabel("Predicted", fontsize=font_size)
        ax_conf.set_ylabel("True", fontsize=font_size)
        
        plt.tight_layout()
        
        # Save output
        output_image_svg = os.path.join(s3_output_dir, f"S3_{final_symbol}_combined.svg")
        output_image_jpg = os.path.join(s3_output_dir, f"S3_{final_symbol}_combined.jpg")
        plt.savefig(output_image_svg, format='svg')
        plt.savefig(output_image_jpg, format='jpg', dpi=300)
        plt.close()
        print(f"Saved: {output_image_svg} and {output_image_jpg}")
    
    # Process all clusters
    if not os.path.exists(plot_data_dir):
        print(f"Plot data directory not found: {plot_data_dir}")
        return
    
    processed_count = 0
    for file in os.listdir(plot_data_dir):
        if file.endswith("_train_roc_data.json"):
            # Extract cluster name from filename "cv_ABC_train_roc_data.json"
            cluster_name = file.replace("_train_roc_data.json", "")
            plot_combined_cluster(cluster_name)
            processed_count += 1
    
    print(f"Generated {processed_count} combined plots for S3 in {s3_output_dir}")


def generate_supplementary_s4_cv_performance(
    all_fold_roc_aucs: list,
    all_fold_f1_scores: list,
    save_dir: str,
    font_size: int = 16
) -> None:
    """
    Generate Supplementary Figure S4: Cross-Validation Performance Scatter Plots.
    
    Creates two scatter plots:
    1. ROC-AUC across random seeds and folds
    2. F1 Score across random seeds and folds
    
    Parameters
    ----------
    all_fold_roc_aucs : list
        List of ROC-AUC values for each seed and fold
    all_fold_f1_scores : list
        List of F1 score values for each seed and fold
    save_dir : str
        Directory to save the output figures directly (e.g., reports_paper/Sub)
    font_size : int, optional
        Font size for plots (default: 16)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Save directly in the specified directory (not in S4 subfolder)
    os.makedirs(save_dir, exist_ok=True)
    
    # --- Drawing scatter plot of ROC-AUC ---
    plt.figure(figsize=(12, 6))
    
    # List for scatter plot (x-axis: seed number, y-axis: ROC-AUC of each fold)
    seeds_scatter = []
    roc_auc_scatter = []
    
    for seed, roc_auc_list in enumerate(all_fold_roc_aucs):
        for auc_value in roc_auc_list:
            if not np.isnan(auc_value):  # Plot excluding NaN
                seeds_scatter.append(seed)
                roc_auc_scatter.append(auc_value)
    
    plt.scatter(seeds_scatter, roc_auc_scatter, alpha=0.6, label="ROC-AUC per Fold")
    
    if roc_auc_scatter:
        # Overall mean ROC-AUC and standard deviation
        overall_mean = np.mean(roc_auc_scatter)
        overall_std = np.std(roc_auc_scatter)
        
        # Draw horizontal line of overall mean with red dashed line
        plt.axhline(y=overall_mean, color='r', linestyle='--', linewidth=2, label="Overall Mean ROC-AUC")
        
        # Notation of "Mean ± Standard Deviation" below the mean line
        mean_std_text = f"{overall_mean:.3f} ± {overall_std:.3f}"
        plt.text(min(seeds_scatter) + 1, overall_mean - 0.07, mean_std_text, color='r', fontsize=font_size)
    
    plt.xlabel("Random Seed", fontsize=font_size)
    plt.ylabel("ROC-AUC", fontsize=font_size)
    plt.title("9-Fold CV ROC-AUC Across Random Seeds", fontsize=font_size+2)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()
    
    scatter_filename_svg = os.path.join(save_dir, "S4_cl_model_9fold_roc_auc_scatter.svg")
    scatter_filename_jpg = os.path.join(save_dir, "S4_cl_model_9fold_roc_auc_scatter.jpg")
    plt.savefig(scatter_filename_svg, dpi=300)
    plt.savefig(scatter_filename_jpg, dpi=300)
    print(f"Saved: {scatter_filename_svg} and {scatter_filename_jpg}")
    plt.close()
    
    # --- Drawing scatter plot of F1 score ---
    plt.figure(figsize=(12, 6))
    
    # List for scatter plot (x-axis: seed number, y-axis: F1 Score of each fold)
    seeds_scatter_f1 = []
    f1_scatter = []
    
    for seed, f1_list in enumerate(all_fold_f1_scores):
        for f1_value in f1_list:
            seeds_scatter_f1.append(seed)
            f1_scatter.append(f1_value)
    
    plt.scatter(seeds_scatter_f1, f1_scatter, alpha=0.6, label="F1 Score per Fold")
    
    if f1_scatter:
        overall_f1_mean = np.mean(f1_scatter)
        overall_f1_std = np.std(f1_scatter)
        
        # Draw horizontal line of overall mean F1 Score with blue dashed line
        plt.axhline(y=overall_f1_mean, color='b', linestyle='--', linewidth=2, label="Overall Mean F1 Score")
        
        # Display text below the mean line
        mean_std_text_f1 = f"{overall_f1_mean:.3f} ± {overall_f1_std:.3f}"
        plt.text(min(seeds_scatter_f1) + 1, overall_f1_mean - 0.07, mean_std_text_f1, color='b', fontsize=font_size)
    
    plt.xlabel("Random Seed", fontsize=font_size)
    plt.ylabel("F1 Score", fontsize=font_size)
    plt.title("9-Fold CV F1 Score Across Random Seeds", fontsize=font_size+2)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()
    
    f1_scatter_filename_svg = os.path.join(save_dir, "S4_cl_model_9fold_f1_score_scatter.svg")
    f1_scatter_filename_jpg = os.path.join(save_dir, "S4_cl_model_9fold_f1_score_scatter.jpg")
    plt.savefig(f1_scatter_filename_svg, dpi=300)
    plt.savefig(f1_scatter_filename_jpg, dpi=300)
    print(f"Saved: {f1_scatter_filename_svg} and {f1_scatter_filename_jpg}")
    plt.close()
    
    print(f"Generated 2 scatter plots for S4 in {save_dir}")


def generate_supplementary_s1_resistance_by_polymer(
    valid_info_cluster: pd.DataFrame,
    polymer_name_dict: dict,
    special_labels: list = None,
    symbol_order: list = None,
    save_dir: str = None,
    image_zoom: float = 0.25
) -> tuple:
    """
    Generate Supplementary Figure S1: Resistance Binary Distribution by Polymer with Structure Images.
    
    This function creates two versions of resistance distribution plots:
    1. Rate version (is_rate=True): Shows resistance ratio
    2. Count version (is_rate=False): Shows data count
    
    Each plot includes chemical structure images for each polymer.
    Polymer labels are colored based on crystallinity (black for crystalline, red for non-crystalline).
    This follows the same approach as Figure 3 for both coloring and ordering.
    
    Parameters
    ----------
    valid_info_cluster : pd.DataFrame
        DataFrame with clustered validation information and resistance data
    polymer_name_dict : dict
        Dictionary mapping SMILES to polymer names with Japanese abbreviations
    special_labels : list, optional
        List of non-crystalline polymer symbols to be colored red
        (e.g., ['PVA', 'PIB', 'PMMA', 'PVdC', 'PC', 'PVC', 'PS', 'PSF', 'PMP'])
    symbol_order : list, optional
        Order of polymer symbols (same as Figure 3's symbol_order from df_sorted)
        If None, will sort by resistance_binary ratio
    save_dir : str, optional
        Directory to save the figures
    image_zoom : float, optional
        Zoom factor for structure images (default: 0.25)
    
    Returns
    -------
    tuple
        (fig_rate, fig_count) Matplotlib Figure objects for rate and count versions
    """
    # Count data for title
    n_total = len(valid_info_cluster)
    n_solvents = valid_info_cluster['smiles_solvent'].nunique()
    n_polymers = valid_info_cluster['smiles_polymer'].nunique()
    title_str = f"Chemical resistance data, total number : {n_total:,}, solvent : {n_solvents}, polymer : {n_polymers}"
    
    # === Rate version (resistance ratio) ===
    fig_rate, ax_rate, ticklabels_rate, keys_rate = _plot_resistance_binary_dists(
        valid_info_cluster,
        polymer_name_dict,
        "smiles_polymer",
        is_rate=True,
        is_reverse=True,
        is_rotate=False,
        show_data_labels=True,
        symbol_order=symbol_order
    )
    
    ax_rate.set_title(title_str)
    
    add_structure_images(
        ax_rate,
        keys_rate,
        polymer_name_dict,
        get_rdkit_image,
        image_zoom=image_zoom,
        label_x=-0.08,
        image_x_offset=0.04,
        image_y_offset=0.0,
        special_labels=special_labels
    )
    
    # === Count version (data count) ===
    fig_count, ax_count, ticklabels_count, keys_count = _plot_resistance_binary_dists(
        valid_info_cluster,
        polymer_name_dict,
        "smiles_polymer",
        is_rate=False,
        is_reverse=True,
        is_rotate=False,
        show_data_labels=True,
        symbol_order=symbol_order
    )
    
    ax_count.set_title(title_str)
    
    add_structure_images(
        ax_count,
        keys_count,
        polymer_name_dict,
        get_rdkit_image,
        image_zoom=image_zoom,
        label_x=-0.08,
        image_x_offset=0.04,
        image_y_offset=0.0,
        special_labels=special_labels
    )
    
    # Save if directory provided
    if save_dir:
        import os
        # Save rate version
        save_as_png(fig_rate, os.path.join(save_dir, "S1_resistance_binary_rates_in_resin_no_cluster_rate.svg"))
        save_as_png(fig_rate, os.path.join(save_dir, "S1_resistance_binary_rates_in_resin_no_cluster_rate.jpg"))
        
        # Save count version
        save_as_png(fig_count, os.path.join(save_dir, "S1_resistance_binary_rates_in_resin_no_cluster_count.svg"))
        save_as_png(fig_count, os.path.join(save_dir, "S1_resistance_binary_rates_in_resin_no_cluster_count.jpg"))
        
        print(f"Supplementary Figure S1 (rate and count versions) saved to {save_dir}")
    
    return fig_rate, fig_count


# =============================================================================
# Supplementary Figure S5: Force Field Parameter Average Plots
# =============================================================================

def generate_supplementary_s5_ff_parameter_analysis(
    df_vPolymer: pd.DataFrame,
    FF_solvent_cols: list,
    save_dir: str,
    base_font_size: int = 14,
    Th_high: float = 0.9,
    Th_low: float = 0.1,
    alpha: float = 0.2
) -> tuple[Figure, Figure]:
    """
    Generate Supplementary Figure S5: Force Field Parameter Average Plots.
    
    Creates two multi-panel figures showing average force field parameters
    for resistant vs non-resistant polymers with ±1σ bands.
    
    Parameters
    ----------
    df_vPolymer : pd.DataFrame
        DataFrame containing resistance predictions and force field features
    FF_solvent_cols : list
        List of force field feature column names
    save_dir : str
        Directory to save the output figures
    base_font_size : int, default=14
        Base font size for plot text
    Th_high : float, default=0.9
        High threshold for non-resistant classification
    Th_low : float, default=0.1
        Low threshold for resistant classification
    alpha : float, default=0.2
        Transparency for ±1σ fill regions
        
    Returns
    -------
    tuple[Figure, Figure]
        (fig_4panel, fig_6panel): The two generated figures
        
    Notes
    -----
    - Figure 1 (4-panel, 2x2): mass, r0, charge, polar
    - Figure 2 (6-panel, 3x2): epsilon, k_angle, sigma, theta0, k_bond, k_dih
    - Saves as SVG and JPG files directly in save_dir
    - Files: S5_subplot_4_Average_with_Std.svg/jpg, S5_subplot_6_Average_with_Std.svg/jpg
    """
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    # Font settings
    title_font_size = base_font_size * 1.2
    xlabel_font_size = base_font_size * 1.1
    ylabel_font_size = base_font_size * 1.1
    tick_font_size = base_font_size
    tick_font_size_y = int(base_font_size * 0.9)
    legend_font_size = base_font_size * 0.9
    
    plt.rcParams.update({'font.family': 'Arial', 'font.size': base_font_size})
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Arial'
    plt.rcParams['mathtext.it'] = 'Arial:italic'
    plt.rcParams['mathtext.bf'] = 'Arial:bold'
    
    # Helper functions
    def get_ff_label(prefix: str) -> str:
        """Get LaTeX label for force field parameter prefix."""
        if prefix.startswith("FF_mass_"):
            return r"$\mathrm{mass}$"
        elif prefix.startswith("FF_charge_"):
            return r"$\mathrm{charge}$"
        elif prefix.startswith("FF_epsilon_"):
            return r"$\epsilon$"
        elif prefix.startswith("FF_sigma_"):
            return r"$\sigma$"
        elif prefix.startswith("FF_k_bond_"):
            return r"$K_{bond}$"
        elif prefix.startswith("FF_r0_"):
            return r"$r_0$"
        elif prefix.startswith("FF_polar_"):
            return r"$\mathrm{polar}$"
        elif prefix.startswith("FF_k_angle_"):
            return r"$K_{angle}$"
        elif prefix.startswith("FF_theta0_"):
            return r"$\theta_0$"
        elif prefix.startswith("FF_k_dih_"):
            return r"$K_{dihedral}$"
        else:
            return r"$Other$"
    
    def remove_ff_prefix(label: str) -> str:
        """Remove FF_ prefix and _solvent suffix from label."""
        if label.startswith("FF_"):
            label = label[len("FF_"):]
        if label.endswith("_solvent"):
            label = label[:-len("_solvent")]
        return label
    
    def set_custom_yticks_fixed(ax):
        """Set y-axis ticks with lower limit at -0.05."""
        _, cur_ymax = ax.get_ylim()
        lower = -0.05
        upper = np.ceil(cur_ymax * 10) / 10
        ticks = np.arange(0, upper + 0.1, 0.1)
        tick_labels = [f"{tick:.1f}" for tick in ticks]
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels, fontsize=tick_font_size_y)
        ax.set_ylim(lower, upper)
    
    def plot_average(ax, prefix: str, df_class1_prob: pd.Series, 
                    vpolymer_features: pd.DataFrame, Th_high: float, Th_low: float,
                    alpha: float = 0.2, legend_outside: bool = False):
        """Plot average force field parameters with ±1σ bands."""
        # Filter by prefix
        df_FFD = vpolymer_features.filter(regex=f"^{prefix}")
        df_FFD_prob1 = df_FFD[df_class1_prob >= Th_high]   # y=1 High probability
        df_FFD_prob0 = df_FFD[df_class1_prob <= Th_low]     # y=0 Low probability
        
        n1, n0 = len(df_FFD_prob1), len(df_FFD_prob0)
        new_labels = [remove_ff_prefix(col) for col in df_FFD.columns]
        
        # Calculate mean and std
        m1, s1 = df_FFD_prob1.mean(0), df_FFD_prob1.std(0)
        m0, s0 = df_FFD_prob0.mean(0), df_FFD_prob0.std(0)
        
        x = np.arange(len(df_FFD.columns))
        
        # Resistant line and ±1σ
        line_R, = ax.plot(x, m0, color='green', lw=2,
                         label=f"Resistant (p<{Th_low}, n={n0:,})")
        ax.fill_between(x, m0 - s0, m0 + s0, color='green', alpha=alpha, label="_nolegend_")
        
        # Non-resistant line and ±1σ
        line_NR, = ax.plot(x, m1, color='red', lw=2,
                          label=f"Non-resistant (p>{Th_high}, n={n1:,})")
        ax.fill_between(x, m1 - s1, m1 + s1, color='red', alpha=alpha, label="_nolegend_")
        
        # Appearance settings
        ax.set_xticks(x)
        ax.set_xticklabels(new_labels, rotation=90, fontsize=tick_font_size)
        set_custom_yticks_fixed(ax)
        ax.set_ylabel("Force field kernel mean descriptor", fontsize=ylabel_font_size)
        ax.set_title(f"Average plot of {get_ff_label(prefix)}", fontsize=title_font_size)
        
        # Legend
        sigma_patch = Patch(facecolor='0.5', alpha=alpha, edgecolor='none')
        if legend_outside:
            ax.legend([line_R, line_NR, sigma_patch],
                     [line_R.get_label(), line_NR.get_label(), "±1σ"],
                     loc='center left', bbox_to_anchor=(1.02, 0.5),
                     frameon=False, handlelength=2.0, labelspacing=0.5, borderaxespad=0.0)
        else:
            ax.legend([line_R, line_NR, sigma_patch],
                     [line_R.get_label(), line_NR.get_label(), "±1σ"],
                     loc='upper right', frameon=False, handlelength=2.0, labelspacing=0.5)
    
    # Prepare data
    df_class1_prob = df_vPolymer['resistance_pred']
    vpolymer_features = df_vPolymer[FF_solvent_cols]
    
    # Create save directory (directly in save_dir, not in subfolder)
    os.makedirs(save_dir, exist_ok=True)
    
    # -------------------------------------------------
    # Figure 1: 4 subplots (2x2)
    # Top Left: mass, Top Right: r0,
    # Bottom Left: charge, Bottom Right: polar
    # -------------------------------------------------
    prefixes_4 = ['FF_mass_', 'FF_r0_', 'FF_charge_', 'FF_polar_']
    fig_4panel, axes_4 = plt.subplots(2, 2, figsize=(12, 10))
    axes_4 = axes_4.flatten()
    
    for i, prefix in enumerate(prefixes_4):
        plot_average(axes_4[i], prefix, df_class1_prob, vpolymer_features, Th_high, Th_low, alpha=alpha)
    
    fig_4panel.tight_layout()
    
    # Save 4-panel figure
    filename_4_svg = os.path.join(save_dir, "S5_subplot_4_Average_with_Std.svg")
    filename_4_jpg = os.path.join(save_dir, "S5_subplot_4_Average_with_Std.jpg")
    fig_4panel.savefig(filename_4_svg, dpi=300, bbox_inches="tight")
    fig_4panel.savefig(filename_4_jpg, dpi=300, bbox_inches="tight")
    print(f"✅ S5 (4-panel) saved: {filename_4_svg} and {filename_4_jpg}")
    
    # -------------------------------------------------
    # Figure 2: 6 subplots (3x2)
    # Top Left: epsilon, Top Right: k_angle,
    # Middle Left: sigma, Middle Right: theta0,
    # Bottom Left: k_bond, Bottom Right: k_dih
    # -------------------------------------------------
    prefixes_6 = ['FF_epsilon_', 'FF_k_angle_', 'FF_sigma_', 'FF_theta0_', 'FF_k_bond_', 'FF_k_dih_']
    fig_6panel, axes_6 = plt.subplots(3, 2, figsize=(12, 15))
    axes_6 = axes_6.flatten()
    
    for i, prefix in enumerate(prefixes_6):
        plot_average(axes_6[i], prefix, df_class1_prob, vpolymer_features, Th_high, Th_low, alpha=alpha)
    
    fig_6panel.tight_layout()
    
    # Save 6-panel figure
    filename_6_svg = os.path.join(save_dir, "S5_subplot_6_Average_with_Std.svg")
    filename_6_jpg = os.path.join(save_dir, "S5_subplot_6_Average_with_Std.jpg")
    fig_6panel.savefig(filename_6_svg, dpi=300, bbox_inches="tight")
    fig_6panel.savefig(filename_6_jpg, dpi=300, bbox_inches="tight")
    print(f"✅ S5 (6-panel) saved: {filename_6_svg} and {filename_6_jpg}")
    
    return fig_4panel, fig_6panel


# =============================================================================
# Supplementary Figure S6: Chi Parameter Histogram by Polymer
# =============================================================================

def generate_supplementary_s6_chi_histograms(
    df_dict: dict,
    symbol_order: Sequence[str],
    save_dir: str,
    base_fontsize: int = 10,
    chi_col: str = "chi",
    binary_col: str = "resistance_pred_binary"
) -> tuple[Figure, Figure]:
    """
    Generate Supplementary Figure S6: Chi Parameter Histogram by Polymer.
    
    Creates two multi-panel figures showing χ parameter histograms for each polymer,
    grouped by symbol order (1-12 and 13-27).
    
    Parameters
    ----------
    df_dict : dict
        Dictionary mapping polymer symbols to DataFrames with chi and binary resistance
    symbol_order : Sequence[str]
        Ordered list of polymer symbols
    save_dir : str
        Directory to save the output figures
    base_fontsize : int, default=10
        Base font size for plot text
    chi_col : str, default="chi"
        Column name for chi parameter
    binary_col : str, default="resistance_pred_binary"
        Column name for binary resistance classification
        
    Returns
    -------
    tuple[Figure, Figure]
        (fig_group1, fig_group2): The two generated figures
        
    Notes
    -----
    - Group 1 (4x3 grid): Polymers 1-12 in symbol_order
    - Group 2 (5x3 grid): Polymers 13-27 in symbol_order
    - Saves as SVG and JPG directly in save_dir
    - Files: S6_chi_hist_group1_1to12.svg/jpg, S6_chi_hist_group2_13to27.svg/jpg
    """
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    
    # Set matplotlib font parameters
    plt.rcParams.update({
        'font.size': base_fontsize,
        'axes.titlesize': base_fontsize,
        'axes.labelsize': base_fontsize,
        'xtick.labelsize': base_fontsize,
        'ytick.labelsize': base_fontsize,
        'legend.fontsize': base_fontsize,
    })
    
    def plot_chi_hist_subplot(ax, df, symbol, chi_col="chi", binary_col="resistance_pred_binary",
                              xlim=None, base_fontsize=18):
        """Plot chi histogram for a single polymer on given axes."""
        title_fontsize = base_fontsize * 1.2
        axis_label_fontsize = base_fontsize * 1.2
        tick_fontsize = base_fontsize
        legend_fontsize = base_fontsize
        
        # Data extraction
        chi = df[chi_col]
        binary = df[binary_col]
        
        # Auto-set x-axis limit if not provided
        if xlim is None:
            chi_min, chi_max = chi.min(), chi.max()
            margin = 0.1 * (chi_max - chi_min) if chi_max != chi_min else 1
            xlim = (chi_min - margin, chi_max + margin)
        
        bins = np.linspace(xlim[0], xlim[1], 51)
        
        chi_0 = chi[binary == 0]
        chi_1 = chi[binary == 1]
        n_0 = len(chi_0)
        n_1 = len(chi_1)
        n_all = n_0 + n_1
        
        # Draw histograms
        ax.hist(chi_0, bins=bins, alpha=0.3, label=f"Resistant (n = {n_0:,})",
                density=True, color="green")
        ax.hist(chi_1, bins=bins, alpha=0.3, label=f"Non-resistant (n = {n_1:,})",
                density=True, color="red")
        
        # Remove leading number from symbol for title
        title_symbol = re.sub(r"^\d+_", "", symbol)
        
        # Set title and labels
        ax.set_title(f"Histogram of $\\mathit{{\\chi}}$ for {title_symbol} vs {n_all:,} solvent", 
                    fontsize=title_fontsize)
        ax.set_xlabel(r"$\mathit{\chi}$", fontsize=axis_label_fontsize)
        ax.set_ylabel("Normalized frequency", fontsize=axis_label_fontsize)
        
        # Tick settings
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        
        # X-axis ticks (multiples of 0.5)
        x_min, x_max = ax.get_xlim()
        raw_step = (x_max - x_min) / 5
        temp_step = np.round(raw_step, 1)
        if temp_step == 0:
            temp_step = 0.1
        step = np.ceil(temp_step / 0.5) * 0.5
        
        if x_min >= 0:
            new_x_min = 0.0
        else:
            new_x_min = np.floor(x_min / step) * step
        if x_max <= 0:
            new_x_max = 0.0
        else:
            new_x_max = np.ceil(x_max / step) * step
        
        xticks = np.arange(new_x_min, new_x_max + step/10, step)
        if not any(np.isclose(xticks, 0.0, atol=1e-8)):
            xticks = np.sort(np.append(xticks, 0.0))
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        
        # Y-axis ticks
        y_min, y_max = ax.get_ylim()
        step_y = np.round((y_max - y_min) / 4, 1)
        if step_y == 0:
            step_y = 0.1
        if y_min >= 0:
            new_y_min = 0.0
        else:
            new_y_min = np.floor(y_min / step_y) * step_y
        if y_max <= 0:
            new_y_max = 0.0
        else:
            new_y_max = np.ceil(y_max / step_y) * step_y
        yticks = np.arange(new_y_min, new_y_max + step_y/10, step_y)
        if not any(np.isclose(yticks, 0.0, atol=1e-8)):
            yticks = np.sort(np.append(yticks, 0.0))
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        
        ax.legend(fontsize=legend_fontsize)
        ax.set_xlim(xlim)
    
    # Group polymers by symbol order number (1-12 vs 13-27)
    group1 = []  # Numbers 1-12
    group2 = []  # Numbers 13-27
    
    for symbol, df in df_dict.items():
        final_symbol = get_final_symbol(symbol, symbol_order)
        # Extract leading number if format is "NN_..."
        m = re.match(r"^(\d{2})_", final_symbol)
        if m:
            num = int(m.group(1))
            if 1 <= num <= 12:
                group1.append((num, symbol, df))
            elif 13 <= num <= 27:
                group2.append((num, symbol, df))
    
    # Sort by number
    group1.sort(key=lambda x: x[0])
    group2.sort(key=lambda x: x[0])
    
    # Create output directory (directly in save_dir, not in subfolder)
    os.makedirs(save_dir, exist_ok=True)
    
    # -------------------------------------------------
    # Group 1: 4 rows x 3 cols (symbols 1-12)
    # -------------------------------------------------
    fig_group1 = None
    if group1:
        rows1, cols1 = 4, 3
        fig_group1, axes1 = plt.subplots(rows1, cols1, figsize=(3*4, 4*4))
        axes1 = axes1.flatten()
        
        # Hide all subplots initially
        for ax in axes1:
            ax.set_visible(False)
        
        # Draw each symbol in group1
        for i, (num, symbol, df) in enumerate(group1):
            ax = axes1[i]
            ax.set_visible(True)
            final_symbol = get_final_symbol(symbol, symbol_order)
            # Replace "solv" with "solvent" for display
            display_symbol = final_symbol.replace("solv", "solvent", 1) if final_symbol.startswith("solv") else final_symbol
            plot_chi_hist_subplot(ax, df, display_symbol, chi_col=chi_col,
                                 binary_col=binary_col, xlim=None, base_fontsize=base_fontsize)
        
        fig_group1.tight_layout()
        
        # Save group1 figures
        save_path_svg = os.path.join(save_dir, "S6_chi_hist_group1_1to12.svg")
        save_path_jpg = os.path.join(save_dir, "S6_chi_hist_group1_1to12.jpg")
        fig_group1.savefig(save_path_svg, dpi=300, format="svg", bbox_inches="tight")
        fig_group1.savefig(save_path_jpg, dpi=300, format="jpg", bbox_inches="tight")
        print(f"✅ S6 (Group 1: 1-12) saved: {save_path_svg}")
    
    # -------------------------------------------------
    # Group 2: 5 rows x 3 cols (symbols 13-27)
    # -------------------------------------------------
    fig_group2 = None
    if group2:
        rows2, cols2 = 5, 3
        fig_group2, axes2 = plt.subplots(rows2, cols2, figsize=(3*4, 5*4))
        axes2 = axes2.flatten()
        
        # Hide all subplots initially
        for ax in axes2:
            ax.set_visible(False)
        
        # Draw each symbol in group2
        for i, (num, symbol, df) in enumerate(group2):
            ax = axes2[i]
            ax.set_visible(True)
            final_symbol = get_final_symbol(symbol, symbol_order)
            display_symbol = final_symbol.replace("solv", "solvent", 1) if final_symbol.startswith("solv") else final_symbol
            plot_chi_hist_subplot(ax, df, display_symbol, chi_col=chi_col,
                                 binary_col=binary_col, xlim=None, base_fontsize=base_fontsize)
        
        fig_group2.tight_layout()
        
        # Save group2 figures
        save_path_svg = os.path.join(save_dir, "S6_chi_hist_group2_13to27.svg")
        save_path_jpg = os.path.join(save_dir, "S6_chi_hist_group2_13to27.jpg")
        fig_group2.savefig(save_path_svg, dpi=300, format="svg", bbox_inches="tight")
        fig_group2.savefig(save_path_jpg, dpi=300, format="jpg", bbox_inches="tight")
        print(f"✅ S6 (Group 2: 13-27) saved: {save_path_svg}")
    
    return fig_group1, fig_group2


# =============================================================================
# ADDITIONAL VISUALIZATION FUNCTIONS
# =============================================================================

def plot_chi_hist_by_binary(
    df: pd.DataFrame,
    symbol: str,
    chi_col: str = "chi",
    binary_col: str = "resistance_pred_binary",
    xlim: Optional[tuple] = None,
    save_path: Optional[str] = None,
    font_size: int = 20
) -> plt.Figure:
    """
    Plot chi parameter histogram colored by binary classification.
    
    Args:
        df: DataFrame with chi and binary columns
        symbol: Symbol for title
        chi_col: Column name for chi values
        binary_col: Column name for binary labels
        xlim: X-axis limits (auto-calculated if None)
        save_path: Path to save figure
        font_size: Font size for labels
        
    Returns:
        Matplotlib figure
    """
    tick_fontsize = font_size
    
    chi = df[chi_col]
    binary = df[binary_col]
    
    # Auto-calculate xlim if not provided
    if xlim is None:
        chi_min, chi_max = chi.min(), chi.max()
        margin = 0.1 * (chi_max - chi_min) if chi_max != chi_min else 1
        xlim = (chi_min - margin, chi_max + margin)
    
    bins = np.linspace(xlim[0], xlim[1], 51)
    
    chi_0 = chi[binary == 0]
    chi_1 = chi[binary == 1]
    n_0 = len(chi_0)
    n_1 = len(chi_1)
    n_all = n_0 + n_1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(chi_0, bins=bins, alpha=0.3, label=f"resistant (n={n_0})", 
            density=True, color="green")
    ax.hist(chi_1, bins=bins, alpha=0.3, label=f"non-resistant (n={n_1})", 
            density=True, color="red")
    
    ax.set_title(f"{symbol}", fontsize=font_size)
    ax.set_xlabel("χ", fontsize=font_size)
    ax.set_ylabel("Normalized frequency", fontsize=font_size)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.legend(fontsize=font_size)
    ax.set_xlim(xlim)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        # Also save as JPG
        jpg_path = save_path.replace('.svg', '.jpg').replace('.png', '.jpg')
        if jpg_path != save_path:
            fig.savefig(jpg_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_chi_hist_from_features(
    df_features: pd.DataFrame,
    df_info: pd.DataFrame,
    model,
    all_features: list,
    exclude_cols: Optional[list] = None,
    chi_col: str = "chi",
    binary_col: str = "resistance_binary",
    xlim: tuple = (-0.5, 7),
    save_path: Optional[str] = None,
    font_size: int = 20
) -> plt.Figure:
    """
    Plot chi parameter histogram from feature data.
    
    Args:
        df_features: DataFrame with feature columns
        df_info: DataFrame with label information
        model: Trained model with predict_proba method
        all_features: List of feature column names
        exclude_cols: Columns to exclude
        chi_col: Column name for chi values
        binary_col: Column name for binary labels
        xlim: X-axis limits
        save_path: Path to save figure
        font_size: Font size for labels
        
    Returns:
        Matplotlib figure
    """
    if exclude_cols is None:
        exclude_cols = ['n_atom_radonpy_polymer', 'mol_weight_radonpy_polymer']
    
    features = [col for col in all_features if col not in exclude_cols]
    train_data = df_features.copy()[features]
    X_train_exp = train_data[features]
    train_data["resistance_pred"] = model.predict_proba(X_train_exp)[:, 1]
    
    label = df_info[binary_col]
    
    chi = X_train_exp[chi_col]
    chi_0 = chi[label == 0]
    chi_1 = chi[label == 1]
    
    n_0 = len(chi_0)
    n_1 = len(chi_1)
    n_all = n_0 + n_1
    
    bins = np.linspace(xlim[0], xlim[1], 51)
    tick_fontsize = font_size
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(chi_0, bins=bins, color="green", alpha=0.3, 
            label=f"resistant (n={n_0})", density=True)
    ax.hist(chi_1, bins=bins, color="red", alpha=0.3, 
            label=f"non-resistant (n={n_1})", density=True)
    
    ax.set_title(f"{chi_col} histogram by {binary_col} (Total n={n_all})", fontsize=font_size)
    ax.set_xlabel("χ", fontsize=font_size)
    ax.set_ylabel("Density", fontsize=font_size)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.legend(fontsize=font_size)
    ax.set_xlim(xlim)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    model,
    feature_names: list,
    save_dir: Optional[str] = None,
    top_n: int = 50,
    title: str = "Feature Importance (Top 20)",
    figsize: tuple = (10, 8)
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Visualize feature importance from model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        save_dir: Directory to save figure
        top_n: Number of top features to display
        title: Plot title
        figsize: Figure size
        
    Returns:
        Tuple of (figure, importance_dataframe)
    """
    import seaborn as sns
    
    feature_importances = model.feature_importances_
    
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importances
    }).sort_values(by="importance", ascending=False)
    
    # Display top features
    importance_df_top = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x="importance", y="feature", data=importance_df_top, palette="viridis", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, "feature_importance.jpg")
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        csv_path = os.path.join(save_dir, "feature_importance.csv")
        importance_df.to_csv(csv_path, index=False)
    
    return fig, importance_df


def save_model_results(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    save_dir: str,
    threshold: float = 0.9
) -> Dict[str, float]:
    """
    Save model training results including ROC curve and confusion matrix.
    
    Args:
        model: Trained model with predict_proba method
        X_train: Training features
        y_train: Training labels
        save_dir: Directory to save results
        threshold: Classification threshold
        
    Returns:
        Dictionary with evaluation metrics (f1, auc, fpr, fnr)
    """
    from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
    import seaborn as sns
    
    os.makedirs(save_dir, exist_ok=True)
    
    y_probs = model.predict_proba(X_train)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    
    # ROC Curve
    fpr_arr, tpr_arr, _ = roc_curve(y_train, y_probs)
    roc_auc = auc(fpr_arr, tpr_arr)
    
    # F1 Score
    f1 = f1_score(y_train, y_pred)
    
    # Confusion Matrix
    cm = confusion_matrix(y_train, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_value = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Save ROC Curve
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot(fpr_arr, tpr_arr, label=f"AUC = {roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title(f"ROC AUC Curve\n"
                     f"F1 (th={threshold}): {f1:.3f}, AUC: {roc_auc:.3f}\n"
                     f"FPR: {fpr_value:.3f}, FNR: {fnr_value:.3f}")
    ax_roc.legend(loc="lower right")
    plt.tight_layout()
    fig_roc.savefig(os.path.join(save_dir, "roc_auc_curve.jpg"), dpi=300)
    plt.close(fig_roc)
    
    # Save Confusion Matrix
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    ax_cm.set_title(f"Confusion Matrix\n"
                    f"F1 (th={threshold}): {f1:.3f}, AUC: {roc_auc:.3f}\n"
                    f"FPR: {fpr_value:.3f}, FNR: {fnr_value:.3f}")
    plt.tight_layout()
    fig_cm.savefig(os.path.join(save_dir, "confusion_matrix.jpg"), dpi=300)
    plt.close(fig_cm)
    
    return {
        "f1": f1,
        "auc": roc_auc,
        "fpr": fpr_value,
        "fnr": fnr_value
    }


def visualize_cluster_metrics(
    metrics_dir: str,
    save_dir: str,
    metrics: Optional[list] = None,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Visualize metrics for each cluster label as horizontal bar plots.
    
    Args:
        metrics_dir: Directory containing metric CSV files
        save_dir: Directory to save figures
        metrics: List of metric names to visualize
        figsize: Figure size per metric
        
    Returns:
        Matplotlib figure with all metrics
    """
    import seaborn as sns
    
    if metrics is None:
        metrics = ["micro_f1", "macro_f1", "auc", "accuracy", "fpr", "fnr"]
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    dfs = {}
    for metric in metrics:
        filepath = os.path.join(metrics_dir, f"cluster_labels_{metric}.csv")
        if os.path.exists(filepath):
            dfs[metric] = pd.read_csv(filepath)
    
    # Create subplots
    fig, axes = plt.subplots(len(metrics), 1, figsize=(figsize[0], len(metrics) * figsize[1]))
    if len(metrics) == 1:
        axes = [axes]
    
    ordered_labels = list(map(str, range(1, 11)))
    
    for ax, metric in zip(axes, metrics):
        if metric not in dfs:
            continue
            
        df_opt = dfs[metric]
        df_opt['group_value'] = df_opt['group_value'].astype(str)
        df_opt.set_index("group_value", inplace=True)
        df_opt = df_opt.reindex(ordered_labels).reset_index()
        
        cluster_list = ['solv_cluster_' + str(label) for label in df_opt['group_value'].tolist()]
        values = df_opt[metric].values
        
        color = sns.color_palette('deep')[0]
        bars = ax.barh(cluster_list, values, color=color, alpha=0.5)
        
        mean_value = np.nanmean(values)
        ax.axvline(mean_value, color='red', linestyle='dashed')
        
        ax.set_xlabel(f'Validation {metric.upper()} Values', fontsize=18)
        ax.set_title(f'2 class classification LOOCV {metric.upper()} score (solv_cluster)', 
                    fontsize=20, y=1.02)
        
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                       f'{value:.2f}', va='center', fontsize=14)
        
        ax.text(mean_value - 0.25, ax.get_ylim()[1],
               f'Mean {metric.upper()}= {mean_value:.2f}',
               va='top', ha='center', color='red', fontsize=14)
        
        ax.set_xlim(0, 1.0)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "cluster_metrics_visualization.jpg")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
