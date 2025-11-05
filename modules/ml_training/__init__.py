"""
ML Training Module

This module contains functionality for:
- Model training and evaluation
- Model management (saving, loading, versioning)
- Performance analysis and metrics
"""

from .model_manager import ModelManager
from .backtester import MLBacktester

__all__ = ['ModelManager', 'MLBacktester']