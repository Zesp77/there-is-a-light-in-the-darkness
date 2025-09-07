"""
Base Strategy Class for Lighter Point Farming

Provides a foundation for all scalp trading strategies with common functionality
including signal generation, risk management, and performance tracking.
"""

import numpy as np
import pandas as pd
import talib.abstract as ta
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseScalpStrategy(ABC):
    """
    Abstract base class for all scalping strategies
    """

    # ADD THIS - Required name attribute
    name = 'BaseScalpStrategy'
    
    # Strategy metadata
    timeframe = '5m'
    can_short = True

    # Risk management
    minimal_roi = {
        "0": 0.0025,
        "5": 0.002,
        "15": 0.0015,
        "30": 0.001,
        "60": 0.0005
    }
    stoploss = -0.004
    startup_candle_count = 50

    # ADD THIS METHOD
    def __init__(self):
        self.stats = {
            'signals_generated': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': 0.0
        }

    # ADD THIS METHOD
    def get_common_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Calculate common indicators used across strategies"""
        try:
            # RSI indicators
            dataframe['rsi_7'] = ta.RSI(dataframe, timeperiod=7)
            dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
            dataframe['rsi_21'] = ta.RSI(dataframe, timeperiod=21)
            
            # Bollinger Bands - FIX: Convert to float parameters
            bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
            dataframe['bb_upper'] = bollinger['upperband']
            dataframe['bb_middle'] = bollinger['middleband']
            dataframe['bb_lower'] = bollinger['lowerband']
            dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lower']) / (dataframe['bb_upper'] - dataframe['bb_lower'])
            dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
            
            # MACD
            macd_data = ta.MACD(dataframe)
            dataframe['macd'] = macd_data['macd']
            dataframe['macd_signal'] = macd_data['macdsignal']
            dataframe['macd_hist'] = macd_data['macdhist']
            
            # EMAs
            dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=9)
            dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
            dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
            dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
            
            # Volume indicators
            dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
            dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
            
            # ATR
            dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
            dataframe['atr_percent'] = dataframe['atr'] / dataframe['close']
            
            # Momentum
            dataframe['momentum'] = ta.MOM(dataframe, timeperiod=5)
            dataframe['roc'] = ta.ROC(dataframe, timeperiod=5)
            
            return dataframe
            
        except Exception as e:
            logger.error(f"❌ Error in get_common_indicators: {e}")
            return dataframe


    # Rest of your existing methods...


def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Populate entry signals based on pair-specific strategies"""
    try:
        pair = metadata['pair']
        
        # Initialize entry columns
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        if 'ENA' in pair:
            # ENA: Mean reversion entries
            ena_long_conditions = (
                (dataframe['bb_percent'] < 0.15) &  # Near lower BB
                (dataframe['rsi_7'] < 25) &         # Fast RSI oversold
                (dataframe['rsi_14'] < 35) &        # Standard RSI oversold
                (dataframe['trend_1h'] >= 0) &      # Trend filter
                (dataframe['volume_ratio'] > 1.2) & # Volume confirmation
                (dataframe['atr_percent'] > 0.015) & # Volatility filter
                (dataframe['close'] > dataframe['ema_50'])  # Above key EMA
            )
            
            ena_short_conditions = (
                (dataframe['bb_percent'] > 0.85) &  # Near upper BB
                (dataframe['rsi_7'] > 75) &         # Fast RSI overbought
                (dataframe['rsi_14'] > 65) &        # Standard RSI overbought
                (dataframe['trend_1h'] <= 0) &      # Trend filter
                (dataframe['volume_ratio'] > 1.2) & # Volume confirmation
                (dataframe['atr_percent'] > 0.015) & # Volatility filter
                (dataframe['close'] < dataframe['ema_50'])   # Below key EMA
            )
            
            dataframe.loc[ena_long_conditions, 'enter_long'] = 1
            dataframe.loc[ena_short_conditions, 'enter_short'] = 1
        
        elif 'LINK' in pair:
            # LINK: Bollinger mean reversion
            link_long_conditions = (
                (dataframe['close'] <= dataframe['bb_lower_tight']) &
                (dataframe['bb_percent'] < 0.1) &
                (dataframe['bb_width'] > 0.02) &    # Sufficient volatility
                (dataframe['rsi_14'] < 30) &
                (dataframe['rsi_21'] < 40) &
                (dataframe['macd'] > dataframe['macd_signal']) &
                (dataframe['ema_21'] > dataframe['ema_50']) &
                (dataframe['trend_1h'] >= 0) &
                (dataframe['volume_ratio'] > 1.1)
            )
            
            link_short_conditions = (
                (dataframe['close'] >= dataframe['bb_upper_tight']) &
                (dataframe['bb_percent'] > 0.9) &
                (dataframe['bb_width'] > 0.02) &
                (dataframe['rsi_14'] > 70) &
                (dataframe['rsi_21'] > 60) &
                (dataframe['macd'] < dataframe['macd_signal']) &
                (dataframe['ema_21'] < dataframe['ema_50']) &
                (dataframe['trend_1h'] <= 0) &
                (dataframe['volume_ratio'] > 1.1)
            )
            
            dataframe.loc[link_long_conditions, 'enter_long'] = 1
            dataframe.loc[link_short_conditions, 'enter_short'] = 1
        
        elif 'AAVE' in pair:
            # AAVE: Breakout momentum
            aave_long_conditions = (
                (dataframe['close'] > dataframe['bb_upper']) &
                (dataframe['close'] > dataframe['donchian_upper'].shift(1)) &
                (dataframe['macd_hist'] > 0) &
                (dataframe['macd'] > dataframe['macd_signal']) &
                (dataframe['momentum'] > 0) &
                (dataframe['roc'] > 0.2) &
                (dataframe['volume_ratio'] > 1.5) &
                (dataframe['ema_9'] > dataframe['ema_21']) &
                (dataframe['trend_1h'] > 0) &
                (dataframe['rsi_14'] < 75)
            )
            
            aave_short_conditions = (
                (dataframe['close'] < dataframe['bb_lower']) &
                (dataframe['close'] < dataframe['donchian_lower'].shift(1)) &
                (dataframe['macd_hist'] < 0) &
                (dataframe['macd'] < dataframe['macd_signal']) &
                (dataframe['momentum'] < 0) &
                (dataframe['roc'] < -0.2) &
                (dataframe['volume_ratio'] > 1.5) &
                (dataframe['ema_9'] < dataframe['ema_21']) &
                (dataframe['trend_1h'] < 0) &
                (dataframe['rsi_14'] > 25)
            )
            
            dataframe.loc[aave_long_conditions, 'enter_long'] = 1
            dataframe.loc[aave_short_conditions, 'enter_short'] = 1
        
        return dataframe
        
    except Exception as e:
        logger.error(f"❌ Error in populate_entry_trend: {e}")
        return dataframe

def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Populate exit signals for profit taking and loss cutting"""
    try:
        pair = metadata['pair']
        
        # Initialize exit columns
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        # Universal exit conditions
        universal_long_exit = (
            (dataframe['bb_percent'] > 0.6) |        # Approaching BB middle/upper
            (dataframe['rsi_14'] > 65) |             # RSI normalization
            (dataframe['macd_hist'] < 0) |           # Momentum loss
            (dataframe['volume_ratio'] < 0.8) |      # Volume drying up
            (dataframe['ema_9'] < dataframe['ema_21']) # Trend change
        )
        
        universal_short_exit = (
            (dataframe['bb_percent'] < 0.4) |        # Approaching BB middle/lower
            (dataframe['rsi_14'] < 35) |             # RSI normalization
            (dataframe['macd_hist'] > 0) |           # Momentum loss
            (dataframe['volume_ratio'] < 0.8) |      # Volume drying up
            (dataframe['ema_9'] > dataframe['ema_21']) # Trend change
        )
        
        # Pair-specific exits
        if 'ENA' in pair:
            ena_long_exit = (
                universal_long_exit |
                (dataframe['rsi_7'] > 70) |          # Fast RSI exit
                (dataframe['close'] >= dataframe['bb_middle']) # BB middle target
            )
            
            ena_short_exit = (
                universal_short_exit |
                (dataframe['rsi_7'] < 30) |
                (dataframe['close'] <= dataframe['bb_middle'])
            )
            
            dataframe.loc[ena_long_exit, 'exit_long'] = 1
            dataframe.loc[ena_short_exit, 'exit_short'] = 1
            
        elif 'LINK' in pair:
            link_long_exit = (
                universal_long_exit |
                (dataframe['close'] >= dataframe['bb_middle'])
            )
            
            link_short_exit = (
                universal_short_exit |
                (dataframe['close'] <= dataframe['bb_middle'])
            )
            
            dataframe.loc[link_long_exit, 'exit_long'] = 1
            dataframe.loc[link_short_exit, 'exit_short'] = 1
            
        elif 'AAVE' in pair:
            aave_long_exit = (
                universal_long_exit |
                (dataframe['momentum'] < 0) |        # Momentum reversal
                (dataframe['rsi_14'] > 80)           # Extreme overbought
            )
            
            aave_short_exit = (
                universal_short_exit |
                (dataframe['momentum'] > 0) |
                (dataframe['rsi_14'] < 20)
            )
            
            dataframe.loc[aave_long_exit, 'exit_long'] = 1
            dataframe.loc[aave_short_exit, 'exit_short'] = 1
        
        return dataframe
        
    except Exception as e:
        logger.error(f"❌ Error in populate_exit_trend: {e}")
        return dataframe

    
    def get_trend_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend-related indicators
        
        Args:
            dataframe: DataFrame with basic indicators
            
        Returns:
            DataFrame with trend indicators
        """
        try:
            # Trend determination using EMAs
            dataframe['trend_short'] = np.where(dataframe['ema_9'] > dataframe['ema_21'], 1, -1)
            dataframe['trend_medium'] = np.where(dataframe['ema_21'] > dataframe['ema_50'], 1, -1)
            dataframe['trend_long'] = np.where(dataframe['ema_50'] > dataframe['ema_200'], 1, -1)
            
            # Overall trend score
            dataframe['trend_score'] = (
                dataframe['trend_short'] * 0.5 +
                dataframe['trend_medium'] * 0.3 +
                dataframe['trend_long'] * 0.2
            )
            
            # Trend strength
            dataframe['trend_strength'] = abs(dataframe['trend_score'])
            
            # ADX for trend strength
            dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
            
            return dataframe
            
        except Exception as e:
            logger.error(f"❌ Error calculating trend indicators: {e}")
            return dataframe
    
    def get_volatility_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility indicators
        
        Args:
            dataframe: DataFrame with basic indicators
            
        Returns:
            DataFrame with volatility indicators
        """
        try:
            # Bollinger Band squeeze
            dataframe['bb_squeeze'] = (
                (dataframe['bb_upper'] - dataframe['bb_lower']) < 
                (dataframe['bb_middle'] * 0.02)
            )
            
            # Volatility percentile
            dataframe['volatility_percentile'] = (
                dataframe['atr_percent'].rolling(window=20).rank() / 20
            )
            
            # True Range
            dataframe['true_range'] = ta.TRANGE(dataframe)
            
            return dataframe
            
        except Exception as e:
            logger.error(f"❌ Error calculating volatility indicators: {e}")
            return dataframe
    
    def custom_stoploss(self, pair: str, trade_duration: timedelta, 
                       current_profit: float, **kwargs) -> float:
        """
        Custom stoploss implementation
        
        Args:
            pair: Trading pair
            trade_duration: Time since trade opened
            current_profit: Current profit percentage
            
        Returns:
            Stoploss percentage (negative value)
        """
        # Time-based stoploss tightening
        if trade_duration < timedelta(minutes=5):
            return self.stoploss  # Initial stoploss
        elif trade_duration < timedelta(minutes=15):
            return self.stoploss * 0.8  # Tighten by 20%
        elif trade_duration < timedelta(minutes=30):
            return self.stoploss * 0.6  # Tighten by 40%
        else:
            return self.stoploss * 0.4  # Very tight for scalping
    
    def custom_exit(self, pair: str, current_profit: float, 
                   trade_duration: timedelta, last_candle: dict, **kwargs) -> Optional[str]:
        """
        Custom exit logic
        
        Args:
            pair: Trading pair
            current_profit: Current profit percentage
            trade_duration: Time since trade opened
            last_candle: Latest candle data
            
        Returns:
            Exit reason string or None
        """
        # Quick profit taking for scalping
        if current_profit > 0.005 and trade_duration > timedelta(minutes=1):
            return 'quick_profit'
        
        # Time-based exit for point farming
        if trade_duration > timedelta(minutes=60):
            return 'time_limit_reached'
        
        return None
    
    def leverage(self, pair: str, **kwargs) -> float:
        """
        Calculate leverage for the pair
        
        Args:
            pair: Trading pair
            
        Returns:
            Leverage multiplier
        """
        # Conservative leverage for scalping
        if 'BTC' in pair or 'ETH' in pair:
            return 3.0  # Lower leverage for major pairs
        elif 'ENA' in pair or 'LINK' in pair:
            return 4.0  # Medium leverage
        elif 'AAVE' in pair:
            return 3.0  # Conservative for breakouts
        else:
            return 2.0  # Very conservative for unknown pairs
    
    def validate_signal(self, signal_data: dict) -> bool:
        """
        Validate signal quality before processing
        
        Args:
            signal_data: Signal data dictionary
            
        Returns:
            True if signal is valid
        """
        try:
            # Basic validation
            required_fields = ['pair', 'side', 'price', 'timestamp']
            if not all(field in signal_data for field in required_fields):
                return False
            
            # Price validation
            if signal_data['price'] <= 0:
                return False
            
            # Side validation
            if signal_data['side'] not in ['long', 'short']:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating signal: {e}")
            return False
    
    def update_stats(self, trade_result: dict):
        """
        Update strategy statistics
        
        Args:
            trade_result: Dictionary containing trade results
        """
        try:
            self.stats['signals_generated'] += 1
            
            if trade_result.get('profit', 0) > 0:
                self.stats['successful_trades'] += 1
            else:
                self.stats['failed_trades'] += 1
            
            self.stats['total_profit'] += trade_result.get('profit', 0)
            
        except Exception as e:
            logger.error(f"❌ Error updating stats: {e}")
    
    def get_performance_stats(self) -> dict:
        """
        Get strategy performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            total_trades = self.stats['successful_trades'] + self.stats['failed_trades']
            
            return {
                'name': self.name,
                'total_signals': self.stats['signals_generated'],
                'total_trades': total_trades,
                'successful_trades': self.stats['successful_trades'],
                'failed_trades': self.stats['failed_trades'],
                'win_rate': (self.stats['successful_trades'] / total_trades * 100) if total_trades > 0 else 0,
                'total_profit': self.stats['total_profit'],
                'avg_profit_per_trade': (self.stats['total_profit'] / total_trades) if total_trades > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting performance stats: {e}")
            return {}
