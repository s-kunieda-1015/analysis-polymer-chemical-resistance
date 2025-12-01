#!/usr/bin/env python3
"""
MIC (Maximal Information Coefficient) calculation functions.

This module contains functions for computing MIC scores
to analyze feature importance in chemical resistance analysis.
"""

from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm
import os


def _compute_single_mic(feature: str, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """
    Compute MIC score for a single feature.
    
    Args:
        feature: Feature name
        X_train: Training feature data
        y_train: Target values
        
    Returns:
        MIC score for this feature
    """
    from minepy import MINE
    mine = MINE()
    mine.compute_score(X_train[feature].values, y_train.values)
    return mine.mic()


def calculate_mic_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: Optional[List[str]] = None,
    n_jobs: int = 1
) -> pd.DataFrame:
    """
    Calculate MIC scores for all features.
    
    Args:
        X_train: Training feature data
        y_train: Target values
        feature_names: Optional list of feature names to compute
        n_jobs: Number of parallel jobs (1 = sequential)
        
    Returns:
        DataFrame with feature names and MIC scores, sorted by MIC descending
    """
    # Get feature names
    if feature_names is None:
        feature_names = X_train.columns.tolist()
    
    print(f"📈 Number of features to compute: {len(feature_names)}")
    
    # Calculate MIC scores
    if n_jobs == 1:
        # Sequential computation
        from minepy import MINE
        mic_scores = []
        mine = MINE()
        for feature in tqdm(feature_names, desc="🔄 Computing MIC"):
            mine.compute_score(X_train[feature].values, y_train.values)
            mic_scores.append(mine.mic())
    else:
        # Parallel computation
        print(f"🔄 Computing MIC ({n_jobs} parallel jobs)...")
        mic_scores = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_compute_single_mic)(feature, X_train, y_train)
            for feature in feature_names
        )
    
    # Create DataFrame
    mic_df = pd.DataFrame({
        "feature": feature_names,
        "MIC": mic_scores
    }).sort_values(by="MIC", ascending=False)
    
    return mic_df


def plot_mic_scores(
    mic_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance by MIC",
    figsize: tuple = (10, 8),
    palette: str = "viridis"
) -> plt.Figure:
    """
    Create bar plot of MIC scores.
    
    Args:
        mic_df: DataFrame with 'feature' and 'MIC' columns
        top_n: Number of top features to display
        title: Plot title
        figsize: Figure size
        palette: Color palette
        
    Returns:
        Matplotlib figure
    """
    # Extract top features
    mic_top_df = mic_df.head(top_n)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x="MIC", y="feature", data=mic_top_df, palette=palette, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Maximal Information Coefficient (MIC)")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    
    return fig


def calculate_and_save_mic(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    dir_name: Optional[str] = None,
    top_n: int = 20,
    cols: Optional[List[str]] = None,
    filename: str = "mic_feature_importance_all.jpg",
    n_jobs: int = 1
) -> pd.DataFrame:
    """
    Calculate MIC scores and save results (plot and CSV).
    
    Args:
        model: Trained model (for compatibility, not used in MIC calculation)
        X_train: Training feature data
        y_train: Target values
        dir_name: Directory to save results (None = don't save)
        top_n: Number of top features to display in plot
        cols: Optional list of column names to filter
        filename: Filename for saved plot
        n_jobs: Number of parallel jobs
        
    Returns:
        DataFrame with MIC scores for all features
    """
    print("📊 Starting MIC calculation...")
    
    # Get feature names
    feature_names = X_train.columns.tolist()
    
    # Filter to specific columns if provided
    if cols:
        feature_names = [col for col in feature_names if col in cols]
        X_train = X_train[feature_names]
    
    # Calculate MIC scores
    mic_df = calculate_mic_scores(X_train, y_train, feature_names, n_jobs)
    
    # Create and save plot
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
        # Create plot
        fig = plot_mic_scores(mic_df, top_n=top_n)
        
        # Save image
        img_path = os.path.join(dir_name, filename)
        fig.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"🖼️ Figure saved to {img_path}")
        
        # Save CSV
        csv_name = filename.replace(".jpg", "_mic_scores.csv")
        csv_path = os.path.join(dir_name, csv_name)
        mic_df.to_csv(csv_path, index=False)
        print(f"📄 MIC scores saved to {csv_path}")
    else:
        print("⚠️ Save directory not specified, files will not be saved.")
    
    print("✅ MIC calculation completed.")
    
    return mic_df

