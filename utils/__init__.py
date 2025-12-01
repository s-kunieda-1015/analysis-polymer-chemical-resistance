#!/usr/bin/env python3
"""
Utility modules for paper figure generation.

This package contains modularized components for the chemical resistance
analysis paper:

- config: Configuration constants (paths, parameters, font sizes, colors, etc.)
- plotting: Plotting functions for generating paper figures
- validation: Cross-validation functions for model evaluation
- train: Model training functions for all models
- metrics: Evaluation metrics calculation functions
- mic: MIC (Maximal Information Coefficient) calculation functions

Usage
-----
Import all configuration and plotting functions:
    >>> from utils.config import *
    >>> from utils import plotting
    >>> from utils import validation
    >>> from utils import train
    >>> from utils import metrics
    >>> from utils import mic

Or import specific modules:
    >>> from utils import config
    >>> from utils import plotting
    >>> from utils import validation
    >>> from utils import train
    >>> from utils import metrics
    >>> from utils import mic
    >>> print(config.REPORT_DIR_MAIN)
    >>> fig = plotting.plot_combined_resistance_distributions(...)
    >>> results = validation.run_logocv_validation(...)
    >>> model = train.train_resistance_model(...)
    >>> result = metrics.calculate_metrics(...)
    >>> mic_df = mic.calculate_mic_scores(...)
"""

from . import config
from . import plotting
from . import validation
from . import train
from . import metrics
from . import mic

__all__ = ['config', 'plotting', 'validation', 'train', 'metrics', 'mic']
__version__ = '1.0.0'
