"""
Utilities module for Lighter Point Farming Bot

Contains helper functions, logging configuration, and common utilities
used throughout the application.
"""

__version__ = "1.0.0"
__author__ = "Lighter Point Farming Bot Team"

from .logger import setup_logger
from .helpers import (
    format_currency,
    calculate_percentage_change,
    validate_pair_format,
    safe_divide,
    round_to_precision
)

__all__ = [
    'setup_logger',
    'format_currency',
    'calculate_percentage_change', 
    'validate_pair_format',
    'safe_divide',
    'round_to_precision'
]
