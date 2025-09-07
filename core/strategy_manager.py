"""
Strategy Manager for Lighter Point Farming Bot

Handles dynamic loading, validation, and management of trading strategies.
Provides a clean interface for strategy operations and performance tracking.
"""

import logging
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Type, Any
import sys
import os

from strategies.base_strategy import BaseScalpStrategy
from utils.logger import setup_logger

logger = setup_logger(__name__)

class StrategyManager:
    """
    Manages trading strategies with dynamic loading and validation
    """
    
    def __init__(self, strategies_path: str = "strategies"):
        """
        Initialize strategy manager
        
        Args:
            strategies_path: Path to strategies directory
        """
        self.strategies_path = Path(strategies_path)
        self.loaded_strategies: Dict[str, Type[BaseScalpStrategy]] = {}
        self.strategy_instances: Dict[str, BaseScalpStrategy] = {}
        
        # Ensure strategies path exists
        if not self.strategies_path.exists():
            self.strategies_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created strategies directory: {self.strategies_path}")
        
        # Load available strategies
        self._discover_strategies()
        
        logger.info(f"🎯 Strategy Manager initialized with {len(self.loaded_strategies)} strategies")
    
    def _discover_strategies(self):
        """Discover and load all available strategies"""
        try:
            # Add strategies path to sys.path if not already there
            strategies_abs_path = str(self.strategies_path.absolute().parent)
            if strategies_abs_path not in sys.path:
                sys.path.insert(0, strategies_abs_path)
            
            # Import strategies module to trigger registration
            try:
                import strategies
                if hasattr(strategies, 'AVAILABLE_STRATEGIES'):
                    self.loaded_strategies.update(strategies.AVAILABLE_STRATEGIES)
                    logger.info(f"✅ Loaded {len(strategies.AVAILABLE_STRATEGIES)} strategies from registry")
            except ImportError as e:
                logger.warning(f"⚠️ Could not import strategies module: {e}")
            
            # Also scan for Python files in strategies directory
            self._scan_strategy_files()
            
        except Exception as e:
            logger.error(f"❌ Error discovering strategies: {e}")
    
    def _scan_strategy_files(self):
        """Scan strategies directory for Python files"""
        try:
            for strategy_file in self.strategies_path.glob("*.py"):
                if strategy_file.name.startswith('__'):
                    continue
                
                try:
                    # Import the module
                    module_name = f"strategies.{strategy_file.stem}"
                    spec = importlib.util.spec_from_file_location(module_name, strategy_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Find strategy classes in the module
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if (issubclass(obj, BaseScalpStrategy) and 
                                obj != BaseScalpStrategy and
                                name not in self.loaded_strategies):
                                
                                self.loaded_strategies[name] = obj
                                logger.debug(f"📊 Discovered strategy: {name}")
                
                except Exception as e:
                    logger.warning(f"⚠️ Could not load strategy from {strategy_file}: {e}")
        
        except Exception as e:
            logger.error(f"❌ Error scanning strategy files: {e}")
    
    def list_available_strategies(self) -> List[str]:
        """
        Get list of available strategy names
        
        Returns:
            List of strategy class names
        """
        return list(self.loaded_strategies.keys())
    
    def get_strategy_info(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a strategy
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with strategy information or None
        """
        try:
            if strategy_name not in self.loaded_strategies:
                return None
            
            strategy_class = self.loaded_strategies[strategy_name]
            
            return {
                'name': strategy_name,
                'class_name': strategy_class.__name__,
                'description': strategy_class.__doc__ or "No description available",
                'timeframe': getattr(strategy_class, 'timeframe', '5m'),
                'can_short': getattr(strategy_class, 'can_short', True),
                'stoploss': getattr(strategy_class, 'stoploss', -0.02),
                'minimal_roi': getattr(strategy_class, 'minimal_roi', {}),
                'startup_candle_count': getattr(strategy_class, 'startup_candle_count', 50)
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting strategy info for {strategy_name}: {e}")
            return None
    
    def load_strategy(self, strategy_name: str) -> Optional[BaseScalpStrategy]:
        """
        Load and instantiate a strategy
        
        Args:
            strategy_name: Name of the strategy to load
            
        Returns:
            Strategy instance or None if not found
        """
        try:
            if strategy_name not in self.loaded_strategies:
                logger.error(f"❌ Strategy not found: {strategy_name}")
                return None
            
            # Return cached instance if available
            if strategy_name in self.strategy_instances:
                logger.debug(f"🔄 Returning cached strategy instance: {strategy_name}")
                return self.strategy_instances[strategy_name]
            
            # Create new instance
            strategy_class = self.loaded_strategies[strategy_name]
            strategy_instance = strategy_class()
            
            # Validate the strategy
            if not self._validate_strategy(strategy_instance):
                logger.error(f"❌ Strategy validation failed: {strategy_name}")
                return None
            
            # Cache the instance
            self.strategy_instances[strategy_name] = strategy_instance
            
            logger.info(f"✅ Strategy loaded successfully: {strategy_name}")
            return strategy_instance
            
        except Exception as e:
            logger.error(f"❌ Error loading strategy {strategy_name}: {e}")
            return None
    
    def _validate_strategy(self, strategy: BaseScalpStrategy) -> bool:
        """
        Validate a strategy instance
        
        Args:
            strategy: Strategy instance to validate
            
        Returns:
            True if strategy is valid
        """
        try:
            # Check required methods
            required_methods = [
                'populate_indicators',
                'populate_entry_trend', 
                'populate_exit_trend'
            ]
            
            for method_name in required_methods:
                if not hasattr(strategy, method_name):
                    logger.error(f"❌ Strategy missing required method: {method_name}")
                    return False
                
                method = getattr(strategy, method_name)
                if not callable(method):
                    logger.error(f"❌ Strategy method not callable: {method_name}")
                    return False
            
            # Check required attributes
            required_attributes = [
                'timeframe',
                'stoploss',
                'minimal_roi'
            ]
            
            for attr_name in required_attributes:
                if not hasattr(strategy, attr_name):
                    logger.error(f"❌ Strategy missing required attribute: {attr_name}")
                    return False
            
            # Validate timeframe
            valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
            if strategy.timeframe not in valid_timeframes:
                logger.error(f"❌ Invalid timeframe: {strategy.timeframe}")
                return False
            
            # Validate stoploss
            if not isinstance(strategy.stoploss, (int, float)) or strategy.stoploss >= 0:
                logger.error(f"❌ Invalid stoploss: {strategy.stoploss}")
                return False
            
            # Validate minimal_roi
            if not isinstance(strategy.minimal_roi, dict):
                logger.error(f"❌ Invalid minimal_roi format")
                return False
            
            logger.debug(f"✅ Strategy validation passed: {strategy.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating strategy: {e}")
            return False
    
    def reload_strategy(self, strategy_name: str) -> Optional[BaseScalpStrategy]:
        """
        Reload a strategy (useful for development)
        
        Args:
            strategy_name: Name of strategy to reload
            
        Returns:
            Reloaded strategy instance or None
        """
        try:
            # Remove cached instance
            if strategy_name in self.strategy_instances:
                del self.strategy_instances[strategy_name]
            
            # Rediscover strategies
            self._discover_strategies()
            
            # Load the strategy again
            return self.load_strategy(strategy_name)
            
        except Exception as e:
            logger.error(f"❌ Error reloading strategy {strategy_name}: {e}")
            return None
    
    def get_strategy_performance(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        Get performance statistics for a strategy
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Performance statistics dictionary or None
        """
        try:
            if strategy_name not in self.strategy_instances:
                logger.warning(f"⚠️ Strategy not loaded: {strategy_name}")
                return None
            
            strategy = self.strategy_instances[strategy_name]
            return strategy.get_performance_stats()
            
        except Exception as e:
            logger.error(f"❌ Error getting strategy performance: {e}")
            return None
    
    def update_strategy_stats(self, strategy_name: str, trade_result: Dict[str, Any]):
        """
        Update strategy statistics
        
        Args:
            strategy_name: Name of the strategy
            trade_result: Trade result data
        """
        try:
            if strategy_name in self.strategy_instances:
                strategy = self.strategy_instances[strategy_name]
                strategy.update_stats(trade_result)
            
        except Exception as e:
            logger.error(f"❌ Error updating strategy stats: {e}")
    
    def get_all_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance statistics for all loaded strategies
        
        Returns:
            Dictionary mapping strategy names to performance stats
        """
        try:
            performance_data = {}
            
            for strategy_name in self.strategy_instances:
                stats = self.get_strategy_performance(strategy_name)
                if stats:
                    performance_data[strategy_name] = stats
            
            return performance_data
            
        except Exception as e:
            logger.error(f"❌ Error getting all strategy performance: {e}")
            return {}
    
    def validate_all_strategies(self) -> Dict[str, bool]:
        """
        Validate all loaded strategies
        
        Returns:
            Dictionary mapping strategy names to validation results
        """
        try:
            validation_results = {}
            
            for strategy_name, strategy_class in self.loaded_strategies.items():
                try:
                    # Create temporary instance for validation
                    temp_instance = strategy_class()
                    validation_results[strategy_name] = self._validate_strategy(temp_instance)
                except Exception as e:
                    logger.error(f"❌ Error validating {strategy_name}: {e}")
                    validation_results[strategy_name] = False
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Error validating all strategies: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup strategy manager resources"""
        try:
            self.strategy_instances.clear()
            self.loaded_strategies.clear()
            logger.info("🧹 Strategy manager cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
