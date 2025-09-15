"""
Extremes Analyzer Module for Statistical Analysis of Price Extremes.

This module provides tools for detecting and analyzing local price extremes
(minima/maxima) in cryptocurrency trading data, with focus on extremes
followed by significant price movements (≥3 USDT).
"""

__version__ = "1.0.0"

from .analyzer import ExtremesAnalyzer
from .extremes_detector import ExtremesDetector
from .visualizer import ExtremesVisualizer

__all__ = ["ExtremesAnalyzer", "ExtremesDetector", "ExtremesVisualizer"]