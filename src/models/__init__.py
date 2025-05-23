# src/__init__.py
"""
Stock LNN Analysis Package
Main package for Liquid Neural Network stock analysis system.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Make key components easily accessible
from .data import StockDataLoader, StockDataPreprocessor
from .models import LiquidNetwork, ModelConfig
from .utils import StockPredictionMetrics, ExperimentTracker

__all__ = [
    'StockDataLoader',
    'StockDataPreprocessor', 
    'LiquidNetwork',
    'ModelConfig',
    'StockPredictionMetrics',
    'ExperimentTracker'
]

---FILE_SEPARATOR---

# src/data/__init__.py
"""
Data handling modules for stock market data.
"""

from .data_loader import StockDataLoader
from .preprocessor import StockDataPreprocessor, FeatureEngineer, prepare_model_data

__all__ = [
    'StockDataLoader',
    'StockDataPreprocessor',
    'FeatureEngineer', 
    'prepare_model_data'
]

---FILE_SEPARATOR---

# src/models/__init__.py
"""
Neural network models for stock prediction.
"""

from .lnn_model import (
    LiquidTimeCell,
    LiquidNetwork, 
    ModelConfig,
    create_sequences,
    count_parameters,
    get_model_summary
)

__all__ = [
    'LiquidTimeCell',
    'LiquidNetwork',
    'ModelConfig', 
    'create_sequences',
    'count_parameters',
    'get_model_summary'
]

---FILE_SEPARATOR---

# src/analysis/__init__.py
"""
Analysis modules for market intelligence and feature engineering.
"""

from .pattern_recognition import PatternRecognizer
from .feature_engineering import AdvancedFeatureEngineer, SimpleFeatureEngineer
from .dimensionality_reduction import DimensionalityReducer, QuickDimensionalityReducer
from .temporal_analysis import TemporalAnalyzer

__all__ = [
    'PatternRecognizer',
    'AdvancedFeatureEngineer',
    'SimpleFeatureEngineer',
    'DimensionalityReducer', 
    'QuickDimensionalityReducer',
    'TemporalAnalyzer'
]

---FILE_SEPARATOR---

# src/utils/__init__.py
"""
Utility modules for metrics, tracking, and visualization.
"""

from .metrics import StockPredictionMetrics, MetricTracker
from .experiment_tracker import ExperimentTracker, log_experiment_simple

__all__ = [
    'StockPredictionMetrics',
    'MetricTracker',
    'ExperimentTracker',
    'log_experiment_simple'
]

---FILE_SEPARATOR---

# config/__init__.py
"""
Configuration management.
"""

# This can stay empty - just makes it a package

---FILE_SEPARATOR---

# scripts/__init__.py
"""
Executable scripts for training, evaluation, and analysis.
"""

# This can stay empty - just makes it a package

---FILE_SEPARATOR---

# tests/__init__.py
"""
Test modules for the stock analysis system.
"""

# This can stay empty for now

---FILE_SEPARATOR---

# notebooks/__init__.py
"""
Jupyter notebooks for exploration and prototyping.
"""

# This can stay empty

---FILE_SEPARATOR---

# data/__init__.py
"""
Data storage directory.
"""

# This can stay empty

---FILE_SEPARATOR---

# results/__init__.py
"""
Results storage directory.
"""

# This can stay empty
