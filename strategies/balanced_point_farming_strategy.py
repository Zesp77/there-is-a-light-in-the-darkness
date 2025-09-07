"""
Balanced Point Farming Strategy for Lighter.xyz

Combines ENA Mean Reversion + LINK Bollinger + Multi-Pair Rotation
Target: 82 trades/day, 0.16% average profit, ~$0.65/point cost

Author: Lighter Point Farming Bot
Optimized for: ENA, LINK, AAVE with 3x+ leverage
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
    Balanced Point Farming Strategy for Lighter.xyz
    Combines ENA Mean Reversion + LINK Bollinger + Multi-Pair Rotation
    """

    # ADD THIS LINE - Required name attribute
    name = 'BalancedPointFarmingStrategy'
    
    # Strategy settings
    timeframe = '5m'
    can_short = True
    minimal_roi = {
        "0": 0.0025,
        "5": 0.002,
        "15": 0.0015,
        "30": 0.001,
        "60": 0.0005
    }
    stoploss = -0.004
    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.002
    startup_candle_count = 50

    # ADD THIS METHOD
    def __init__(self):
        super().__init__()
        self.stats = {
            'signals_generated': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': 0.0
        }

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Populate indicators for balanced point farming strategy"""
        
        # Get pair name for adaptive indicators
        pair = metadata['pair']
        
        # RSI indicators
        dataframe['rsi_7'] = ta.RSI(dataframe, timeperiod=7)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_21'] = ta.RSI(dataframe, timeperiod=21)
        
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_middle'] = bollinger['middleband'] 
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lower']) / (dataframe['bb_upper'] - dataframe['bb_lower'])
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        
        # Tight Bollinger Bands
        bollinger_tight = ta.BBANDS(dataframe, timeperiod=20, nbdevup=1.5, nbdevdn=1.5)
        dataframe['bb_upper_tight'] = bollinger_tight['upperband']
        dataframe['bb_lower_tight'] = bollinger_tight['lowerband']
        
        # MACD
        macd_data = ta.MACD(dataframe)
        dataframe['macd'] = macd_data['macd']
        dataframe['macd_signal'] = macd_data['macdsignal']
        dataframe['macd_hist'] = macd_data['macdhist']
        
        # EMAs
        dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        
        # Volume indicators
        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_percent'] = dataframe['atr'] / dataframe['close']
        
        # Momentum
        dataframe['momentum'] = ta.MOM(dataframe, timeperiod=5)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=5)
        
        # Pair-specific indicators
        if 'ENA' in pair:
            dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=5)
            dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=5)
        elif 'LINK' in pair:
            dataframe['sma_20'] = ta.SMA(dataframe, timeperiod=20)
            dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=50)
        elif 'AAVE' in pair:
            dataframe['donchian_upper'] = ta.MAX(dataframe['high'], timeperiod=20)
            dataframe['donchian_lower'] = ta.MIN(dataframe['low'], timeperiod=20)
        
        # Add trend from higher timeframe (simulate 1h trend)
        dataframe['trend_1h'] = np.where(dataframe['ema_50'] > dataframe['ema_50'].shift(12), 1, -1)
        
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Populate entry signals based on pair-specific strategies"""
        
        pair = metadata['pair']
        
        # Initialize entry columns
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        if 'ENA' in pair:
            # ENA Long entries
            ena_long = (
                (dataframe['bb_percent'] < 0.15) &
                (dataframe['rsi_7'] < 25) &
                (dataframe['rsi_14'] < 35) &
                (dataframe['trend_1h'] >= 0) &
                (dataframe['volume_ratio'] > 1.2) &
                (dataframe['atr_percent'] > 0.015) &
                (dataframe['close'] > dataframe['ema_50'])
            )
            
            # ENA Short entries
            ena_short = (
                (dataframe['bb_percent'] > 0.85) &
                (dataframe['rsi_7'] > 75) &
                (dataframe['rsi_14'] > 65) &
                (dataframe['trend_1h'] <= 0) &
                (dataframe['volume_ratio'] > 1.2) &
                (dataframe['atr_percent'] > 0.015) &
                (dataframe['close'] < dataframe['ema_50'])
            )
            
            dataframe.loc[ena_long, 'enter_long'] = 1
            dataframe.loc[ena_short, 'enter_short'] = 1
            
        elif 'LINK' in pair:
            # LINK Long entries
            link_long = (
                (dataframe['close'] <= dataframe['bb_lower_tight']) &
                (dataframe['bb_percent'] < 0.1) &
                (dataframe['bb_width'] > 0.02) &
                (dataframe['rsi_14'] < 30) &
                (dataframe['rsi_21'] < 40) &
                (dataframe['macd'] > dataframe['macd_signal']) &
                (dataframe['ema_21'] > dataframe['ema_50']) &
                (dataframe['trend_1h'] >= 0) &
                (dataframe['volume_ratio'] > 1.1)
            )
            
            # LINK Short entries
            link_short = (
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
            
            dataframe.loc[link_long, 'enter_long'] = 1
            dataframe.loc[link_short, 'enter_short'] = 1
            
        elif 'AAVE' in pair:
            # AAVE Long entries (breakouts)
            aave_long = (
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
            
            # AAVE Short entries (breakdowns)
            aave_short = (
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
            
            dataframe.loc[aave_long, 'enter_long'] = 1
            dataframe.loc[aave_short, 'enter_short'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Populate exit signals for profit taking and loss cutting"""
        
        pair = metadata['pair']
        
        # Initialize exit columns
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        # Universal exit conditions
        universal_long_exit = (
            (dataframe['bb_percent'] > 0.6) |
            (dataframe['rsi_14'] > 65) |
            (dataframe['macd_hist'] < 0) |
            (dataframe['volume_ratio'] < 0.8) |
            (dataframe['ema_9'] < dataframe['ema_21'])
        )
        
        universal_short_exit = (
            (dataframe['bb_percent'] < 0.4) |
            (dataframe['rsi_14'] < 35) |
            (dataframe['macd_hist'] > 0) |
            (dataframe['volume_ratio'] < 0.8) |
            (dataframe['ema_9'] > dataframe['ema_21'])
        )
        
        # Pair-specific exits
        if 'ENA' in pair:
            ena_long_exit = (
                universal_long_exit |
                (dataframe['rsi_7'] > 70) |
                (dataframe['close'] >= dataframe['bb_middle'])
            )
            ena_short_exit = (
                universal_short_exit |
                (dataframe['rsi_7'] < 30) |
                (dataframe['close'] <= dataframe['bb_middle'])
            )
            dataframe.loc[ena_long_exit, 'exit_long'] = 1
            dataframe.loc[ena_short_exit, 'exit_short'] = 1
            
        elif 'LINK' in pair:
            link_long_exit = universal_long_exit | (dataframe['close'] >= dataframe['bb_middle'])
            link_short_exit = universal_short_exit | (dataframe['close'] <= dataframe['bb_middle'])
            dataframe.loc[link_long_exit, 'exit_long'] = 1
            dataframe.loc[link_short_exit, 'exit_short'] = 1
            
        elif 'AAVE' in pair:
            aave_long_exit = (
                universal_long_exit |
                (dataframe['momentum'] < 0) |
                (dataframe['rsi_14'] > 80)
            )
            aave_short_exit = (
                universal_short_exit |
                (dataframe['momentum'] > 0) |
                (dataframe['rsi_14'] < 20)
            )
            dataframe.loc[aave_long_exit, 'exit_long'] = 1
            dataframe.loc[aave_short_exit, 'exit_short'] = 1
        
        return dataframe