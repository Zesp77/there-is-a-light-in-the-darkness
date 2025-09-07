"""
BALANCED Balanced Point Farming Strategy for Lighter.xyz

FIXED VERSION:
- More achievable entry conditions
- Better signal generation across all pairs
- Realistic performance expectations
- Improved entry/exit logic
"""

import numpy as np
import pandas as pd
import talib.abstract as ta
from datetime import datetime, timedelta
from typing import Dict, List
import logging
from strategies.base_strategy import BaseScalpStrategy

logger = logging.getLogger(__name__)

class BalancedPointFarmingStrategy(BaseScalpStrategy):
    """
    BALANCED Balanced Point Farming Strategy - Quality signals with reasonable frequency
    """

    name = 'BalancedPointFarmingStrategy'

    # Strategy settings - BALANCED
    timeframe = '5m'
    can_short = True
    minimal_roi = {
        "0": 0.012,    # 1.2% immediate profit target
        "10": 0.008,   # 0.8% after 10 minutes
        "20": 0.005,   # 0.5% after 20 minutes
        "30": 0.003    # 0.3% after 30 minutes
    }

    stoploss = -0.008  # 0.8% stop loss
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.008
    startup_candle_count = 50

    def __init__(self):
        super().__init__()
        self.stats = {
            'signals_generated': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': 0.0
        }

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Calculate all required indicators"""
        
        # RSI indicators
        dataframe['rsi_9'] = ta.RSI(dataframe, timeperiod=9)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_21'] = ta.RSI(dataframe, timeperiod=21)

        # Bollinger Bands - FIXED: Use floats
        bb_20 = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_upper'] = bb_20['upperband']
        dataframe['bb_middle'] = bb_20['middleband']
        dataframe['bb_lower'] = bb_20['lowerband']
        
        # Handle division by zero for bb_percent
        bb_range = dataframe['bb_upper'] - dataframe['bb_lower']
        bb_range = np.where(bb_range == 0, 0.0001, bb_range)
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lower']) / bb_range
        dataframe['bb_width'] = bb_range / dataframe['bb_middle']

        # Moving Averages
        dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)

        # MACD
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        dataframe['macd_hist'] = macd['macdhist']

        # Volume analysis
        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = np.where(
            dataframe['volume_sma'] > 0,
            dataframe['volume'] / dataframe['volume_sma'],
            1.0
        )

        # Volatility
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_percent'] = np.where(
            dataframe['close'] > 0,
            dataframe['atr'] / dataframe['close'],
            0.01
        )

        # Momentum
        dataframe['mom_10'] = ta.MOM(dataframe, timeperiod=10)
        dataframe['roc_10'] = ta.ROC(dataframe, timeperiod=10)

        # Stochastic
        stoch = ta.STOCH(dataframe)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']

        # Williams %R
        dataframe['williams_r'] = ta.WILLR(dataframe, timeperiod=14)

        # Trend simulation (1h equivalent)
        dataframe['trend_1h'] = np.where(
            (dataframe['ema_50'] > dataframe['ema_50'].shift(12)) & 
            (dataframe['close'] > dataframe['ema_50']), 1, 
            np.where(
                (dataframe['ema_50'] < dataframe['ema_50'].shift(12)) & 
                (dataframe['close'] < dataframe['ema_50']), -1, 0
            )
        )

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """BALANCED entry signals - Achievable conditions"""
        
        pair = metadata['pair']
        
        # Initialize entry columns
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        # BASE CONDITIONS - More achievable
        base_volume = dataframe['volume_ratio'] > 0.8  # Reduced from 1.1
        base_volatility = dataframe['atr_percent'] > 0.003  # Reduced from 0.005
        
        if 'ENA' in pair:
            # ENA: Mean Reversion Strategy - MORE ACHIEVABLE
            
            # Long conditions - RELAXED
            ena_long = (
                # RSI oversold (relaxed)
                (
                    (dataframe['rsi_14'] < 40) |  # Standard oversold
                    (dataframe['rsi_9'] < 35)     # Fast RSI oversold
                ) &
                
                # Near lower Bollinger Band (relaxed)
                (dataframe['bb_percent'] < 0.3) &  # Was 0.2
                
                # MACD showing potential reversal OR positive momentum
                (
                    (dataframe['macd'] > dataframe['macd_signal']) |
                    (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1))
                ) &
                
                # Base conditions
                base_volume & base_volatility
            )

            # Short conditions - RELAXED
            ena_short = (
                # RSI overbought (relaxed)
                (
                    (dataframe['rsi_14'] > 60) |  # Standard overbought (was 65)
                    (dataframe['rsi_9'] > 65)     # Fast RSI overbought
                ) &
                
                # Near upper Bollinger Band (relaxed)
                (dataframe['bb_percent'] > 0.7) &  # Was 0.8
                
                # MACD showing potential reversal OR negative momentum
                (
                    (dataframe['macd'] < dataframe['macd_signal']) |
                    (dataframe['macd_hist'] < dataframe['macd_hist'].shift(1))
                ) &
                
                # Base conditions
                base_volume & base_volatility
            )

            dataframe.loc[ena_long, 'enter_long'] = 1
            dataframe.loc[ena_short, 'enter_short'] = 1

        elif 'LINK' in pair:
            # LINK: Bollinger Band + RSI Strategy - MORE ACHIEVABLE
            
            # Long conditions - RELAXED
            link_long = (
                # RSI conditions (relaxed)
                (
                    (dataframe['rsi_14'] < 45) |  # Was 40
                    ((dataframe['rsi_14'] > 20) & (dataframe['rsi_14'] < 40))  # Added range
                ) &
                
                # Bollinger conditions (multiple triggers)
                (
                    (dataframe['close'] <= dataframe['bb_lower'] * 1.01) |  # Touch lower BB (relaxed)
                    (dataframe['bb_percent'] < 0.25) |  # Low in BB range
                    (dataframe['williams_r'] < -80)  # Williams %R oversold
                ) &
                
                # Stochastic oversold (relaxed)
                (dataframe['stoch_k'] < 35) &  # Was 30
                
                # Base conditions
                base_volume & base_volatility
            )

            # Short conditions - RELAXED
            link_short = (
                # RSI conditions (relaxed)
                (
                    (dataframe['rsi_14'] > 55) |  # Was 60
                    ((dataframe['rsi_14'] > 60) & (dataframe['rsi_14'] < 80))  # Added range
                ) &
                
                # Bollinger conditions (multiple triggers)
                (
                    (dataframe['close'] >= dataframe['bb_upper'] * 0.99) |  # Touch upper BB (relaxed)
                    (dataframe['bb_percent'] > 0.75) |  # High in BB range
                    (dataframe['williams_r'] > -20)  # Williams %R overbought
                ) &
                
                # Stochastic overbought (relaxed)
                (dataframe['stoch_k'] > 65) &  # Was 70
                
                # Base conditions
                base_volume & base_volatility
            )

            dataframe.loc[link_long, 'enter_long'] = 1
            dataframe.loc[link_short, 'enter_short'] = 1

        elif 'AAVE' in pair:
            # AAVE: Momentum/Breakout Strategy - MUCH MORE ACHIEVABLE
            
            # Long conditions - VERY RELAXED
            aave_long = (
                # Momentum conditions (multiple options)
                (
                    (dataframe['mom_10'] > 0) |  # Basic momentum
                    (dataframe['roc_10'] > 0.2) |  # Rate of change
                    (dataframe['macd_hist'] > 0) |  # MACD histogram positive
                    (dataframe['close'] > dataframe['close'].shift(3))  # Simple price momentum
                ) &
                
                # RSI in reasonable zone (very wide)
                (
                    (dataframe['rsi_14'] > 35) &  # Not oversold (was 45)
                    (dataframe['rsi_14'] < 75)    # Not extreme overbought (was 70)
                ) &
                
                # Volume condition (relaxed)
                (dataframe['volume_ratio'] > 0.9) &  # Was 1.3
                
                # Base volatility
                base_volatility
            )

            # Short conditions - VERY RELAXED
            aave_short = (
                # Momentum conditions (multiple options)
                (
                    (dataframe['mom_10'] < 0) |  # Basic momentum
                    (dataframe['roc_10'] < -0.2) |  # Rate of change
                    (dataframe['macd_hist'] < 0) |  # MACD histogram negative
                    (dataframe['close'] < dataframe['close'].shift(3))  # Simple price momentum
                ) &
                
                # RSI in reasonable zone (very wide)
                (
                    (dataframe['rsi_14'] < 65) &  # Not overbought (was 55)
                    (dataframe['rsi_14'] > 25)    # Not extreme oversold (was 30)
                ) &
                
                # Volume condition (relaxed)
                (dataframe['volume_ratio'] > 0.9) &  # Was 1.3
                
                # Base volatility
                base_volatility
            )

            dataframe.loc[aave_long, 'enter_long'] = 1
            dataframe.loc[aave_short, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """BALANCED exit signals"""
        
        pair = metadata['pair']
        
        # Initialize exit columns
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        # Universal exit conditions - BALANCED
        universal_long_exit = (
            # RSI based exits
            (dataframe['rsi_14'] > 75) |  # Strong overbought (was 70)
            
            # Bollinger Band exits
            (dataframe['close'] >= dataframe['bb_middle']) |  # Back to middle
            
            # MACD reversal
            (
                (dataframe['macd'] < dataframe['macd_signal']) &
                (dataframe['macd_hist'] < dataframe['macd_hist'].shift(1))
            ) |
            
            # Quick profit take
            (dataframe['close'] > dataframe['close'].shift(2) * 1.005) |  # 0.5% profit (was 0.6%)
            
            # Williams %R exit
            (dataframe['williams_r'] > -10)  # Very overbought
        )

        universal_short_exit = (
            # RSI based exits
            (dataframe['rsi_14'] < 25) |  # Strong oversold (was 30)
            
            # Bollinger Band exits
            (dataframe['close'] <= dataframe['bb_middle']) |  # Back to middle
            
            # MACD reversal
            (
                (dataframe['macd'] > dataframe['macd_signal']) &
                (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1))
            ) |
            
            # Quick profit take
            (dataframe['close'] < dataframe['close'].shift(2) * 0.995) |  # 0.5% profit
            
            # Williams %R exit
            (dataframe['williams_r'] < -90)  # Very oversold
        )

        # Apply universal exits
        dataframe.loc[universal_long_exit, 'exit_long'] = 1
        dataframe.loc[universal_short_exit, 'exit_short'] = 1

        return dataframe