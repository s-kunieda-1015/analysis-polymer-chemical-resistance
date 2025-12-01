#!/usr/bin/env python3
"""
Metrics calculation functions for chemical resistance analysis.

This module contains evaluation metric functions for:
1. Classification metrics (confusion matrix, F1, AUC, etc.)
2. Model result saving
3. Group-wise metric calculation
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
import os


# =============================================================================
# Classification Metrics
# =============================================================================

def calculate_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_probs: np.ndarray,
    num_classes: int
) -> Dict[str, Any]:
    """
    Calculate classification metrics for model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities
        num_classes: Number of classes
        
    Returns:
        Dictionary containing:
        - confusion_matrix: Confusion matrix array
        - classwise_f1: F1 score for each class
        - micro_f1: Micro-averaged F1 score
        - macro_f1: Macro-averaged F1 score
        - auc: ROC-AUC score
        - accuracy: Accuracy score
        - fpr: False Positive Rate (binary only)
        - fnr: False Negative Rate (binary only)
    """
    labels = list(range(num_classes))
    
    # Confusion matrix
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    
    # F1 scores
    classwise_f1 = f1_score(y_true, y_pred, average=None, labels=labels)
    micro_f1 = f1_score(y_true, y_pred, average="micro", labels=labels)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels)
    
    # AUC (use probability as it is not affected by threshold)
    auc_val = roc_auc_score(y_true, y_probs, multi_class='ovr', labels=labels)
    
    # Accuracy
    accuracy_val = accuracy_score(y_true, y_pred)
    
    # FPR, FNR for binary classification
    if matrix.size == 4:
        tn, fp, fn, tp = matrix.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    else:
        fpr = 0
        fnr = 0

    return {
        "confusion_matrix": matrix,
        "classwise_f1": classwise_f1,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "auc": auc_val,
        "accuracy": accuracy_val,
        "fpr": fpr,
        "fnr": fnr
    }


def calculate_group_metrics(
    model,
    X_data: pd.DataFrame,
    y_data: pd.Series,
    group_info: pd.DataFrame,
    group_column: str,
    num_classes: int,
    threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Calculate metrics for each group in the data.
    
    Args:
        model: Trained model with predict_proba method
        X_data: Feature data
        y_data: Target labels
        group_info: DataFrame with group information
        group_column: Column name for grouping
        num_classes: Number of classes
        threshold: Classification threshold
        
    Returns:
        List of dictionaries with metrics for each group
    """
    metrics_list = []
    
    for group_value, group_indices in group_info.groupby(group_column).groups.items():
        X_group = X_data.loc[group_indices]
        y_group = y_data.loc[group_indices]
        
        # Predict
        y_probs = model.predict_proba(X_group)[:, 1]
        y_pred = (y_probs >= threshold).astype(int)
        
        # Calculate metrics
        metrics = calculate_metrics(y_group, y_pred, y_probs, num_classes)
        
        metrics_list.append({
            "group_value": group_value,
            **metrics
        })
    
    return metrics_list


def save_metrics_by_group(
    model,
    X_data: pd.DataFrame,
    y_data: pd.Series,
    group_info: pd.DataFrame,
    group_column: str,
    save_dir: str,
    num_classes: int,
    threshold: float = 0.5,
    new_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate and save AUC, F1, Accuracy, FPR, FNR for each group.
    
    Args:
        model: Trained model with predict_proba method
        X_data: Feature data
        y_data: Target labels
        group_info: DataFrame with group information
        group_column: Column name for grouping
        save_dir: Directory to save results
        num_classes: Number of classes
        threshold: Classification threshold
        new_columns: Optional list to sort rows by
        
    Returns:
        DataFrame with metrics for each group
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate metrics for each group
    metrics_list = calculate_group_metrics(
        model, X_data, y_data, group_info, group_column, num_classes, threshold
    )
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    # Sort by new_columns if provided
    if new_columns is not None:
        ordered_df = metrics_df.set_index('group_value').reindex(new_columns).reset_index()
        remaining_df = metrics_df[~metrics_df['group_value'].isin(new_columns)]
        metrics_df = pd.concat([ordered_df, remaining_df], ignore_index=True)
    
    # Save each metric to separate files
    for metric in ["micro_f1", "macro_f1", "auc", "accuracy", "fpr", "fnr"]:
        metric_cols = ["group_value", metric]
        metric_df = metrics_df[metric_cols].copy()
        
        # Save to Excel and CSV
        excel_filename = os.path.join(save_dir, f"{group_column}_{metric}.xlsx")
        csv_filename = os.path.join(save_dir, f"{group_column}_{metric}.csv")
        
        metric_df.to_excel(excel_filename, index=False)
        metric_df.to_csv(csv_filename, index=False)
    
    return metrics_df


# =============================================================================
# Variance/Standard Deviation Calculations
# =============================================================================

def calc_std_all(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Calculate combined standard deviation from polymer and solvent groups.
    
    Args:
        df: DataFrame with group-level metrics
        metric: Metric name (e.g., 'micro_f1', 'auc')
        
    Returns:
        Updated DataFrame with combined standard deviation
    """
    # Variance for each group
    var_polymer = df[f"all_polymer_std_{metric}"] ** 2
    var_solvent = df[f"all_solvent_std_{metric}"] ** 2
    mean_polymer = df[f"all_polymer_mean_{metric}"]
    mean_solvent = df[f"all_solvent_mean_{metric}"]
    mean_all = df[f"all_mean_{metric}"]
    
    # Between-group variance
    var_between = ((mean_polymer - mean_all) ** 2 + (mean_solvent - mean_all) ** 2) / 2
    
    # Combined variance and standard deviation
    var_all = (var_polymer + var_solvent + var_between) / 2
    std_all = np.sqrt(var_all)
    
    df[f"all_std_{metric}"] = std_all
    df.sort_values(f"all_mean_{metric}", ascending=False, inplace=True)
    df = df.reset_index(drop=True)
    
    return df


# =============================================================================
# Classification Type Utilities
# =============================================================================

def set_classification_type(binary_classification: bool) -> tuple:
    """
    Set classification type based on binary flag.
    
    Args:
        binary_classification: True for binary, False for 4-class
        
    Returns:
        Tuple of (classification_type_string, num_classes)
    """
    if binary_classification:
        return "2_class", 2
    else:
        return "4_class", 4


def set_y_values(
    binary_classification: bool,
    valid_info: pd.DataFrame,
    train_info: pd.DataFrame,
    test_info: pd.DataFrame
) -> tuple:
    """
    Set target values based on classification type.
    
    Args:
        binary_classification: True for binary, False for 4-class
        valid_info: Full dataset info
        train_info: Training dataset info
        test_info: Test dataset info
        
    Returns:
        Tuple of (y_all, y_train, y_test)
    """
    if binary_classification:
        return (
            valid_info["resistance_binary"],
            train_info["resistance_binary"],
            test_info["resistance_binary"]
        )
    else:
        return (
            valid_info["resistance"],
            train_info["resistance"],
            test_info["resistance"]
        )

