#!/usr/bin/env python3
"""
Model training functions for chemical resistance analysis.

This module contains training functions for:
1. Chemical resistance prediction (GBDT)
2. Crystallinity classification (RandomForest)
3. Chi parameter prediction (XGBoost and other regression models)
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.base import ClassifierMixin, RegressorMixin
import joblib
import os


# =============================================================================
# Chemical Resistance Model (GBDT)
# =============================================================================

def train_resistance_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 42
) -> GradientBoostingClassifier:
    """
    Train GBDT model for chemical resistance prediction.
    
    Args:
        X_train: Training feature data
        y_train: Training labels
        params: Hyperparameters for GBDT. If None, uses optimized parameters.
        random_state: Random state for reproducibility
        
    Returns:
        Trained GradientBoostingClassifier model
    """
    # Default optimized parameters
    if params is None:
        params = {
            "learning_rate": 0.0995950237281464,
            "n_estimators": 499,
            "max_depth": 10,
            "min_samples_split": 9,
            "min_samples_leaf": 6,
            "subsample": 0.738723859213826,
            "max_features": None
        }
    
    model = GradientBoostingClassifier(random_state=random_state, **params)
    model.fit(X_train, y_train)
    
    return model


def train_and_save_resistance_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any],
    save_dir: str,
    model_filename: str = "chemical_resistance_model.pkl"
) -> GradientBoostingClassifier:
    """
    Train and save GBDT model for chemical resistance prediction.
    
    Args:
        X_train: Training feature data
        y_train: Training labels
        params: Hyperparameters for GBDT
        save_dir: Directory to save model
        model_filename: Filename for saved model
        
    Returns:
        Trained GradientBoostingClassifier model
    """
    # Train model
    model = train_resistance_model(X_train, y_train, params)
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_filepath = os.path.join(save_dir, model_filename)
    joblib.dump(model, model_filepath)
    
    return model


# =============================================================================
# Crystallinity Model (RandomForest)
# =============================================================================

def train_crystallinity_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    **kwargs
) -> RandomForestClassifier:
    """
    Train RandomForest model for crystallinity classification.
    
    Args:
        X_train: Training feature data
        y_train: Training labels (crystallinity binary)
        random_state: Random state for reproducibility
        **kwargs: Additional parameters for RandomForestClassifier
        
    Returns:
        Trained RandomForestClassifier model
    """
    model = RandomForestClassifier(random_state=random_state, **kwargs)
    model.fit(X_train, y_train)
    
    return model


def train_and_save_crystallinity_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    save_dir: str,
    model_filename: str = "crystallinity_model.pkl",
    random_state: int = 42,
    **kwargs
) -> RandomForestClassifier:
    """
    Train and save RandomForest model for crystallinity classification.
    
    Args:
        X_train: Training feature data
        y_train: Training labels
        save_dir: Directory to save model
        model_filename: Filename for saved model
        random_state: Random state for reproducibility
        **kwargs: Additional parameters for RandomForestClassifier
        
    Returns:
        Trained RandomForestClassifier model
    """
    # Train model
    model = train_crystallinity_model(X_train, y_train, random_state, **kwargs)
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_filepath = os.path.join(save_dir, model_filename)
    joblib.dump(model, model_filepath)
    
    return model


# =============================================================================
# Chi Parameter Model (XGBoost and other regressors)
# =============================================================================

def get_chi_model(model_name: str, random_state: int = 42) -> RegressorMixin:
    """
    Get regression model instance for chi parameter prediction.
    
    Args:
        model_name: Model type ("GBDT", "RandomForest", "SVR", "Ridge", "XGBoost")
        random_state: Random state for reproducibility
        
    Returns:
        Regression model instance
        
    Note:
        n_jobs=1 is set for RandomForest and XGBoost to prevent excessive CPU usage
        when running multiple CVs sequentially (only LOGOCV should use parallelization).
    """
    if model_name == "GBDT":
        return GradientBoostingRegressor(random_state=random_state)
    elif model_name == "RandomForest":
        return RandomForestRegressor(random_state=random_state, n_jobs=1)
    elif model_name == "SVR":
        return SVR()
    elif model_name == "Ridge":
        return Ridge(random_state=random_state)
    elif model_name == "XGBoost":
        return XGBRegressor(random_state=random_state, n_jobs=1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def train_chi_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "XGBoost",
    random_state: int = 42
) -> RegressorMixin:
    """
    Train regression model for chi parameter prediction.
    
    Args:
        X_train: Training feature data
        y_train: Training target values (chi parameter)
        model_name: Model type ("GBDT", "RandomForest", "SVR", "Ridge", "XGBoost")
        random_state: Random state for reproducibility
        
    Returns:
        Trained regression model
    """
    model = get_chi_model(model_name, random_state)
    model.fit(X_train, y_train)
    
    return model


def train_and_save_chi_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    save_dir: str,
    model_name: str = "XGBoost",
    model_filename: str = "chi_parameter_model.pkl",
    random_state: int = 42
) -> RegressorMixin:
    """
    Train and save regression model for chi parameter prediction.
    
    Args:
        X_train: Training feature data
        y_train: Training target values
        save_dir: Directory to save model
        model_name: Model type
        model_filename: Filename for saved model
        random_state: Random state for reproducibility
        
    Returns:
        Trained regression model
    """
    # Train model
    model = train_chi_model(X_train, y_train, model_name, random_state)
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_filepath = os.path.join(save_dir, model_filename)
    joblib.dump(model, model_filepath)
    
    return model


# =============================================================================
# Model Loading
# =============================================================================

def load_model(model_filepath: str) -> Any:
    """
    Load a saved model from file.
    
    Args:
        model_filepath: Path to saved model file
        
    Returns:
        Loaded model
    """
    model = joblib.load(model_filepath)
    return model

