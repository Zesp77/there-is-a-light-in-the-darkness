"""
Core module for Lighter Point Farming Bot

Contains the main bot logic, data providers, signal analysis, and trading execution.
"""

__version__ = "1.0.0"
__author__ = "Lighter Point Farming Bot Team"

from .bot_controller import ScalpBotController
from .signal_analyzer import SignalAnalyzer
from .telegram_notifier import TelegramNotifier
from .strategy_manager import StrategyManager
from .backtesting_engine import BacktestingEngine
from .data_provider import MarketDataProvider

__all__ = [
    'ScalpBotController',
    'SignalAnalyzer', 
    'TelegramNotifier',
    'StrategyManager',
    'BacktestingEngine',
    'MarketDataProvider'
]
