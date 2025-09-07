"""
Enhanced Strategies module for Lighter Point Farming Bot

Contains all trading strategies with a modular architecture for easy addition
and modification of trading algorithms.
"""

__version__ = "1.1.0"
__author__ = "Lighter Point Farming Bot Team"

# Import available strategies
from .balanced_point_farming_strategy import BalancedPointFarmingStrategy
from .enhanced_single_trade_strategy import EnhancedSingleTradeStrategy
from .base_strategy import BaseScalpStrategy

# Enhanced strategy registry for dynamic loading
AVAILABLE_STRATEGIES = {
    'BalancedPointFarmingStrategy': BalancedPointFarmingStrategy,
    'EnhancedSingleTradeStrategy': EnhancedSingleTradeStrategy,
    'BaseScalpStrategy': BaseScalpStrategy,
}

def get_strategy_class(strategy_name: str):
    """Get strategy class by name"""
    return AVAILABLE_STRATEGIES.get(strategy_name)

def list_available_strategies():
    """Get list of all available strategy names"""
    return list(AVAILABLE_STRATEGIES.keys())
