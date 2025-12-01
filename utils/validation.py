#!/usr/bin/env python3
"""
Model validation functions for chemical resistance analysis.

This module contains cross-validation functions for:
1. Chemical resistance prediction (LOGOCV)
2. Crystallinity classification (9-fold CV with multiple seeds)
3. Chi parameter prediction (10-fold CV)
"""

from typing import Tuple, Dict, List, Any, Optional, Callable
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, auc,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import os


# =============================================================================
# LOGOCV for Chemical Resistance
# =============================================================================

def _process_single_fold_logocv(
    cv_index: int,
    train_index: np.ndarray,
    val_index: np.ndarray,
    train_data: pd.DataFrame,
    train_info: pd.DataFrame,
    y_train: pd.Series,
    group_column: str,
    cv_prefix: str,
    num_classes: int,
    threshold: float,
    iloc_index: Optional[int],
    train_cv_func: Callable,
) -> pd.DataFrame:
    """Process a single fold for LOGOCV cross-validation (for parallel execution).
    
    Args:
        cv_index: Index of the current fold
        train_index: Indices for training data
        val_index: Indices for validation data
        train_data: Training feature data
        train_info: Training metadata with group information
        y_train: Target labels
        group_column: Column name for grouping
        cv_prefix: Prefix for saved files
        num_classes: Number of classes
        threshold: Classification threshold
        iloc_index: Optional index for iloc access
        train_cv_func: Training function that performs train/val split
        
    Returns:
        DataFrame with validation results
    """
    group_value = train_info.iloc[val_index, iloc_index] if iloc_index is not None else train_info.iloc[val_index][group_column]
    # Use first element if group_value is Series
    group_str = group_value.iloc[0] if hasattr(group_value, "iloc") else group_value
    print(f"Processing {group_str}")
    df = train_cv_func(
        f"{cv_prefix}_{group_str}",
        train_data.iloc[train_index],
        y_train.iloc[train_index],
        train_data.iloc[val_index],
        y_train.iloc[val_index],
        num_classes,
        threshold
    )
    return df


def run_logocv_validation(
    train_data: pd.DataFrame,
    train_info: pd.DataFrame,
    y_train: pd.Series,
    group_column: str,
    cv_prefix: str,
    dfs: List[pd.DataFrame],
    num_classes: int,
    threshold: float = 0.5,
    iloc_index: Optional[int] = None,
    n_jobs: int = 1,
    train_cv_func: Optional[Callable] = None,
):
    """
    Run Leave-One-Group-Out Cross-Validation for chemical resistance prediction.
    
    Args:
        train_data: Training feature data
        train_info: Training metadata with group information
        y_train: Target labels
        group_column: Column name for grouping (e.g., 'resin_cluster' or 'solvent_cluster')
        cv_prefix: Prefix for saved files (e.g., 'resin' or 'solvent')
        dfs: List to append validation results to
        num_classes: Number of classes
        threshold: Classification threshold
        iloc_index: Optional index for iloc access
        n_jobs: Number of parallel jobs
        train_cv_func: Training function that performs train/val split
    """
    # Fold split by LeaveOneGroupOut
    folds = list(LeaveOneGroupOut().split(train_data, groups=train_info[group_column]))
    
    if n_jobs == 1:
        # Sequential processing (original behavior)
        for cv_index, (train_index, val_index) in enumerate(folds):
            df = _process_single_fold_logocv(
                cv_index, train_index, val_index, train_data, train_info,
                y_train, group_column, cv_prefix, num_classes, threshold, iloc_index,
                train_cv_func
            )
            dfs.append(df)
    else:
        # Parallel processing
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_process_single_fold_logocv)(
                cv_index, train_index, val_index, train_data, train_info,
                y_train, group_column, cv_prefix, num_classes, threshold, iloc_index,
                train_cv_func
            )
            for cv_index, (train_index, val_index) in enumerate(folds)
        )
        dfs.extend(results)


# =============================================================================
# 9-fold CV for Crystallinity Classification
# =============================================================================

def _train_single_seed_cv(seed: int, X: pd.DataFrame, y: pd.Series, n_splits: int = 9) -> Tuple:
    """Train crystallinity model with a single random seed using k-fold CV.
    
    Args:
        seed: Random seed for this iteration
        X: Feature matrix
        y: Target labels
        n_splits: Number of folds for cross-validation
        
    Returns:
        Tuple of (mean_accuracy, mean_precision, mean_recall, mean_f1, fold_roc_aucs, fold_f1_scores)
    """
    # Convert y to 1D array to avoid DataConversionWarning
    y_1d = y.values.ravel() if hasattr(y, 'values') else np.ravel(y)
    
    # Model with this seed
    model = RandomForestClassifier(random_state=42)
    
    # Stratified k-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Lists for saving evaluation metrics
    accuracies, precisions, recalls, f1_scores, roc_aucs = [], [], [], [], []
    
    for train_idx, test_idx in skf.split(X, y_1d):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_1d[train_idx], y_1d[test_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # Probability of Class 1
        
        # Calculation of evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Calculation of ROC-AUC (only if y_test contains both class 0 and 1)
        if len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            roc_auc = np.nan
        
        # Save scores
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        roc_aucs.append(roc_auc)
    
    # Return mean metrics and fold-level ROC-AUC and F1 scores
    return (np.mean(accuracies), np.mean(precisions), np.mean(recalls), 
            np.mean(f1_scores), roc_aucs, f1_scores)


def run_crystallinity_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_iterations: int = 100,
    n_splits: int = 9,
    n_jobs: int = 8,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Run Stratified k-Fold Cross-Validation with multiple random seeds for crystallinity.
    
    Args:
        X: Feature matrix
        y: Target labels (crystallinity binary)
        n_iterations: Number of random seeds to test
        n_splits: Number of folds for cross-validation
        n_jobs: Number of parallel jobs
        random_state: Base random state for reproducibility
        
    Returns:
        Dictionary containing:
        - 'accuracies': List of mean accuracies for each seed
        - 'precisions': List of mean precisions
        - 'recalls': List of mean recalls
        - 'f1_scores': List of mean F1 scores
        - 'roc_aucs': List of mean ROC-AUC scores
        - 'all_fold_roc_aucs': List of fold-level ROC-AUC scores for each seed
        - 'all_fold_f1_scores': List of fold-level F1 scores for each seed
    """
    # Parallel execution of CV for each seed (verbose=0 to suppress output)
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_train_single_seed_cv)(seed, X, y, n_splits=n_splits)
        for seed in range(n_iterations)
    )
    
    # Unpack results
    all_accuracies = [r[0] for r in results]
    all_precisions = [r[1] for r in results]
    all_recalls = [r[2] for r in results]
    all_f1_scores = [r[3] for r in results]
    all_fold_roc_aucs = [r[4] for r in results]  # List of fold-level ROC-AUCs for each seed
    all_fold_f1_scores = [r[5] for r in results]  # List of fold-level F1 scores for each seed
    
    # Calculate statistical information of ROC-AUC excluding NaN
    all_roc_aucs = [
        np.mean([x for x in roc_aucs if not np.isnan(x)]) 
        if any(not np.isnan(x) for x in roc_aucs) 
        else np.nan 
        for roc_aucs in all_fold_roc_aucs
    ]
    
    return {
        'accuracies': all_accuracies,
        'precisions': all_precisions,
        'recalls': all_recalls,
        'f1_scores': all_f1_scores,
        'roc_aucs': all_roc_aucs,
        'all_fold_roc_aucs': all_fold_roc_aucs,
        'all_fold_f1_scores': all_fold_f1_scores
    }


# =============================================================================
# 10-fold CV for Chi Parameter Prediction
# =============================================================================

# =============================================================================
# LOGOCV Training and Evaluation Utilities
# =============================================================================

def _calculate_cv_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    num_classes: int
) -> Dict[str, Any]:
    """
    Calculate metrics for CV fold evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities
        num_classes: Number of classes
        
    Returns:
        Dictionary with evaluation metrics
    """
    labels = list(range(num_classes))
    
    # Confusion matrix
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    
    # F1 scores
    classwise_f1 = f1_score(y_true, y_pred, average=None, labels=labels)
    micro_f1 = f1_score(y_true, y_pred, average="micro", labels=labels)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels)
    
    # AUC
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


def _save_confusion_matrix(cm, labels, filename, title, metrics):
    """
    Save Confusion Matrix figure.
    
    Args:
        cm: Confusion matrix array
        labels: Class labels
        filename: Save path
        title: Figure title
        metrics: Dictionary with evaluation metrics
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f"{title}\n"
              f"micro_f1: {metrics['micro_f1']:.3f}, macro_f1: {metrics['macro_f1']:.3f}\n"
              f"auc: {metrics['auc']:.3f}, accuracy: {metrics['accuracy']:.3f}\n"
              f"fpr: {metrics['fpr']:.3f}, fnr: {metrics['fnr']:.3f}")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def _save_roc_curve(y_true, y_probs, num_classes, filename, title, metrics):
    """
    Save ROC Curve figure.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        num_classes: Number of classes
        filename: Save path
        title: Figure title
        metrics: Dictionary with evaluation metrics
    """
    plt.figure(figsize=(8, 6))
    if num_classes == 2:
        if y_probs.ndim > 1:
            probs = y_probs[:, 1]
        else:
            probs = y_probs
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc_val:.3f})")
    else:
        if y_probs.ndim == 1:
            y_probs = np.stack([1 - y_probs, y_probs], axis=1)
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i])
            roc_auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc_val:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    num_samples = len(y_true)
    plt.title(f"{title} (Samples: {num_samples})\n"
              f"micro_f1: {metrics['micro_f1']:.3f}, macro_f1: {metrics['macro_f1']:.3f}\n"
              f"auc: {metrics['auc']:.3f}, accuracy: {metrics['accuracy']:.3f}\n"
              f"fpr: {metrics['fpr']:.3f}, fnr: {metrics['fnr']:.3f}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def train_cv_fold(
    val_cluster: str,
    x_train: pd.DataFrame,
    y_train_true: pd.Series,
    x_val: pd.DataFrame,
    y_val_true: pd.Series,
    num_classes: int,
    threshold: float,
    params: Dict[str, Any],
    dir_name: str
) -> pd.DataFrame:
    """
    Train model for a single CV fold and save results.
    
    Args:
        val_cluster: Name of validation cluster/group
        x_train: Training features
        y_train_true: Training labels
        x_val: Validation features
        y_val_true: Validation labels
        num_classes: Number of classes
        threshold: Classification threshold
        params: Model hyperparameters
        dir_name: Directory to save results
        
    Returns:
        DataFrame with validation metrics
    """
    # Train model
    model = GradientBoostingClassifier(random_state=42, **params)
    model.fit(x_train, y_train_true)
    model.input_features_ = x_train.columns.tolist()

    # Get probabilities
    y_train_probs = model.predict_proba(x_train)[:, 1]
    y_val_probs = model.predict_proba(x_val)[:, 1]

    # Generate predicted labels based on threshold
    y_train_pred = (y_train_probs >= threshold).astype(int)
    y_val_pred = (y_val_probs >= threshold).astype(int)

    # Calculate metrics
    train_metrics = _calculate_cv_metrics(y_train_true, y_train_pred, y_train_probs, num_classes)
    val_metrics = _calculate_cv_metrics(y_val_true, y_val_pred, y_val_probs, num_classes)
    
    # Create directories
    cm_dir = os.path.join(dir_name, 'confusion_matrix')
    roc_dir = os.path.join(dir_name, 'roc_curve')
    plot_data_dir = os.path.join(dir_name, 'plot_data')
    os.makedirs(cm_dir, exist_ok=True)
    os.makedirs(roc_dir, exist_ok=True)
    os.makedirs(plot_data_dir, exist_ok=True)

    # Save confusion matrices
    train_cm_filename = os.path.join(cm_dir, f"{val_cluster}_train_confusion_matrix.png")
    _save_confusion_matrix(train_metrics["confusion_matrix"], list(range(num_classes)),
                          train_cm_filename, f"Train - {val_cluster}", train_metrics)
    val_cm_filename = os.path.join(cm_dir, f"{val_cluster}_val_confusion_matrix.png")
    _save_confusion_matrix(val_metrics["confusion_matrix"], list(range(num_classes)),
                          val_cm_filename, f"Validation - {val_cluster}", val_metrics)
    
    # Save ROC curves
    val_roc_filename = os.path.join(roc_dir, f"{val_cluster}_val_roc_curve.png")
    _save_roc_curve(y_val_true, y_val_probs, num_classes, val_roc_filename,
                   f"Validation ROC - {val_cluster}", val_metrics)
    train_roc_filename = os.path.join(roc_dir, f"{val_cluster}_train_roc_curve.png")
    _save_roc_curve(y_train_true, y_train_probs, num_classes, train_roc_filename,
                   f"Train ROC - {val_cluster}", train_metrics)
    
    # Save confusion matrix data
    train_confusion_file = os.path.join(plot_data_dir, f"{val_cluster}_train_confusion_data.csv")
    val_confusion_file = os.path.join(plot_data_dir, f"{val_cluster}_val_confusion_data.csv")
    pd.DataFrame(train_metrics["confusion_matrix"]).to_csv(train_confusion_file, index=False)
    pd.DataFrame(val_metrics["confusion_matrix"]).to_csv(val_confusion_file, index=False)
    
    # Save ROC curve raw data (validation)
    roc_data = {}
    if num_classes == 2:
        probs_val = y_val_probs[:, 1] if y_val_probs.ndim > 1 else y_val_probs
        fpr_val, tpr_val, thresholds_val = roc_curve(y_val_true, probs_val)
        roc_data = {
            "fpr": fpr_val.tolist(),
            "tpr": tpr_val.tolist(),
            "thresholds": thresholds_val.tolist(),
            "auc": auc(fpr_val, tpr_val)
        }
    roc_data_file = os.path.join(plot_data_dir, f"{val_cluster}_val_roc_data.json")
    with open(roc_data_file, "w") as f:
        json.dump(roc_data, f, indent=4,
                  default=lambda x: x.tolist() if hasattr(x, "tolist") 
                                    else float(x) if isinstance(x, np.generic) 
                                    else str(x))
    
    # Save ROC curve raw data (train)
    train_roc_data = {}
    if num_classes == 2:
        probs_train = y_train_probs[:, 1] if y_train_probs.ndim > 1 else y_train_probs
        fpr_train, tpr_train, thresholds_train = roc_curve(y_train_true, probs_train)
        train_roc_data = {
            "fpr": fpr_train.tolist(),
            "tpr": tpr_train.tolist(),
            "thresholds": thresholds_train.tolist(),
            "auc": auc(fpr_train, tpr_train)
        }
    train_roc_data_file = os.path.join(plot_data_dir, f"{val_cluster}_train_roc_data.json")
    with open(train_roc_data_file, "w") as f:
        json.dump(train_roc_data, f, indent=4,
                  default=lambda x: x.tolist() if hasattr(x, "tolist") 
                                    else float(x) if isinstance(x, np.generic) 
                                    else str(x))
    
    # Save metrics
    train_metrics_file = os.path.join(plot_data_dir, f"{val_cluster}_train_metrics.json")
    val_metrics_file = os.path.join(plot_data_dir, f"{val_cluster}_val_metrics.json")
    with open(train_metrics_file, "w") as f:
        json.dump(train_metrics, f, indent=4,
                  default=lambda x: x.tolist() if hasattr(x, "tolist") 
                                    else float(x) if isinstance(x, np.generic) 
                                    else str(x))
    with open(val_metrics_file, "w") as f:
        json.dump(val_metrics, f, indent=4,
                  default=lambda x: x.tolist() if hasattr(x, "tolist") 
                                    else float(x) if isinstance(x, np.generic) 
                                    else str(x))
    
    return pd.DataFrame({
        f"{val_cluster}_micro_f1": [val_metrics['micro_f1']],
        f"{val_cluster}_macro_f1": [val_metrics['macro_f1']],
        f"{val_cluster}_auc": [val_metrics['auc']],
        f"{val_cluster}_accuracy": [val_metrics['accuracy']],
        f"{val_cluster}_fpr": [val_metrics['fpr']],
        f"{val_cluster}_fnr": [val_metrics['fnr']]
    })


def run_chi_parameter_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model: Any,
    n_splits: int = 10,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Run k-Fold Cross-Validation for chi parameter prediction.
    
    Args:
        X: Feature matrix
        y: Target values (chi parameter)
        model: Regression model instance (e.g., XGBRegressor)
        n_splits: Number of folds for cross-validation
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary containing:
        - 'r2_scores': List of R2 scores for each fold
        - 'mae_scores': List of MAE scores for each fold
        - 'rmse_scores': List of RMSE scores for each fold
        - 'cv_results': List of dictionaries with fold-level results
        - 'parity_data': List of dictionaries with y_true, y_pred for plotting
        - 'mean_r2': Mean R2 across folds
        - 'std_r2': Standard deviation of R2
        - 'mean_mae': Mean MAE across folds
        - 'std_mae': Standard deviation of MAE
        - 'mean_rmse': Mean RMSE across folds
        - 'std_rmse': Standard deviation of RMSE
    """
    # Setting for k-fold CV
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Lists for saving evaluation metrics
    r2_scores, mae_scores, rmse_scores = [], [], []
    cv_results = []
    parity_data = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model (create new instance for each fold)
        from sklearn.base import clone
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        
        # Predict
        y_pred = fold_model.predict(X_val)
        
        # Evaluation
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        # Save scores
        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        
        cv_results.append({
            'Fold': fold + 1,
            'R2': r2,
            'MAE': mae,
            'RMSE': rmse
        })
        
        # Save data for plotting
        parity_data.append({
            'y_true': y_val,
            'y_pred': y_pred,
            'title': f'Fold {fold + 1}',
            'r2': r2,
            'mae': mae,
            'rmse': rmse
        })
    
    # Calculate mean and standard deviation
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores, ddof=1)  # Sample standard deviation
    mean_mae = np.mean(mae_scores)
    std_mae = np.std(mae_scores, ddof=1)
    mean_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores, ddof=1)
    
    return {
        'r2_scores': r2_scores,
        'mae_scores': mae_scores,
        'rmse_scores': rmse_scores,
        'cv_results': cv_results,
        'parity_data': parity_data,
        'mean_r2': mean_r2,
        'std_r2': std_r2,
        'mean_mae': mean_mae,
        'std_mae': std_mae,
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse
    }


