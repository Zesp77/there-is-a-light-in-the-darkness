"""
Enhanced Single Trade Point Farming Strategy for Lighter.xyz

FIXED VERSION - Balanced between quality and signal generation:
- Only one trade at a time
- Quality filters but achievable conditions
- Better entry/exit logic for each pair
- Reasonable signal generation frequency
"""

import numpy as np
import pandas as pd
import talib.abstract as ta
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

from strategies.base_strategy import BaseScalpStrategy

logger = logging.getLogger(__name__)

class EnhancedSingleTradeStrategy(BaseScalpStrategy):
    """
    Enhanced Single Trade Strategy - FIXED VERSION
    Generates quality signals with reasonable frequency
    """
    
    name = 'EnhancedSingleTradeStrategy'
    
    # Strategy settings - BALANCED
    timeframe = '5m'
    can_short = True
    minimal_roi = {
        "0": 0.012,    # 1.2% immediate profit target
        "5": 0.008,    # 0.8% after 5 minutes
        "10": 0.005,   # 0.5% after 10 minutes
        "15": 0.003,   # 0.3% after 15 minutes
        "30": 0.002    # 0.2% after 30 minutes
    }
    
    stoploss = -0.008  # 0.8% stop loss
    trailing_stop = True
    trailing_stop_positive = 0.004
    trailing_stop_positive_offset = 0.006
    startup_candle_count = 50  # Reasonable data requirement
    
    def __init__(self):
        super().__init__()
        self.stats = {
            'signals_generated': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': 0.0
        }
        
        # Single trade tracking
        self.current_trade = None
        self.last_signal_time = {}
        self.signal_cooldown = 300  # 5 minutes cooldown (reduced)
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Calculate indicators with reasonable complexity"""
        
        # RSI indicators
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_21'] = ta.RSI(dataframe, timeperiod=21)
        
        # Bollinger Bands
        bb_20 = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_upper'] = bb_20['upperband']
        dataframe['bb_middle'] = bb_20['middleband'] 
        dataframe['bb_lower'] = bb_20['lowerband']
        
        # Bollinger Band metrics
        bb_range = dataframe['bb_upper'] - dataframe['bb_lower']
        bb_range = np.where(bb_range == 0, 0.0001, bb_range)
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lower']) / bb_range
        dataframe['bb_width'] = bb_range / dataframe['bb_middle']
        
        # Moving averages
        dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        
        # MACD
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        dataframe['macd_hist'] = macd['macdhist']
        
        # Volume analysis
        dataframe['volume_sma_20'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = np.where(
            dataframe['volume_sma_20'] > 0,
            dataframe['volume'] / dataframe['volume_sma_20'],
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
        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']
        
        # Williams %R
        dataframe['williams_r'] = ta.WILLR(dataframe, timeperiod=14)
        
        # Trend detection
        dataframe['trend_score'] = np.where(
            (dataframe['ema_9'] > dataframe['ema_21']) & 
            (dataframe['ema_21'] > dataframe['ema_50']), 1,
            np.where(
                (dataframe['ema_9'] < dataframe['ema_21']) & 
                (dataframe['ema_21'] < dataframe['ema_50']), -1, 0
            )
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """FIXED entry signals - Balanced conditions that will generate signals"""
        
        pair = metadata['pair']
        
        # Initialize entry columns
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        # BALANCED base conditions
        good_volume = dataframe['volume_ratio'] > 1.1  # Reduced from 1.8
        decent_volatility = dataframe['atr_percent'] > 0.004  # Reduced from 0.007
        
        if 'ENA' in pair:
            # ENA: BALANCED Mean Reversion Strategy
            ena_long = (
                # RSI oversold (balanced)
                (dataframe['rsi_14'] < 35) &  # Was 25
                
                # Bollinger Band position (relaxed)
                (dataframe['bb_percent'] < 0.25) &  # Was 0.15
                
                # MACD confirmation (simplified)
                (dataframe['macd'] > dataframe['macd_signal']) &
                
                # Base conditions
                good_volume & decent_volatility
            )
            
            ena_short = (
                # RSI overbought (balanced)
                (dataframe['rsi_14'] > 65) &  # Was 75
                
                # Bollinger Band position (relaxed)
                (dataframe['bb_percent'] > 0.75) &  # Was 0.85
                
                # MACD confirmation (simplified)
                (dataframe['macd'] < dataframe['macd_signal']) &
                
                # Base conditions
                good_volume & decent_volatility
            )
            
            dataframe.loc[ena_long, 'enter_long'] = 1
            dataframe.loc[ena_short, 'enter_short'] = 1
        
        elif 'LINK' in pair:
            # LINK: BALANCED Bollinger Band Strategy
            link_long = (
                # RSI conditions (relaxed)
                (dataframe['rsi_14'] < 40) &  # Was 28
                
                # Bollinger conditions (achievable)
                (dataframe['bb_percent'] < 0.2) &  # Was 0.1
                
                # Stochastic oversold (relaxed)
                (dataframe['stoch_k'] < 30) &  # Was 20
                
                # Base conditions
                good_volume & decent_volatility
            )
            
            link_short = (
                # RSI conditions (relaxed)
                (dataframe['rsi_14'] > 60) &  # Was 72
                
                # Bollinger conditions (achievable)
                (dataframe['bb_percent'] > 0.8) &  # Was 0.9
                
                # Stochastic overbought (relaxed)
                (dataframe['stoch_k'] > 70) &  # Was 80
                
                # Base conditions
                good_volume & decent_volatility
            )
            
            dataframe.loc[link_long, 'enter_long'] = 1
            dataframe.loc[link_short, 'enter_short'] = 1
        
        elif 'AAVE' in pair:
            # AAVE: BALANCED Momentum Strategy
            aave_long = (
                # Momentum conditions (achievable)
                (dataframe['mom_10'] > 0) &
                (dataframe['roc_10'] > 0.1) &  # Was 0.5
                
                # RSI in reasonable zone
                (dataframe['rsi_14'] > 40) &  # Was 45
                (dataframe['rsi_14'] < 70) &  # Was 65
                
                # Trend alignment (relaxed)
                (dataframe['trend_score'] >= 0) &  # Was > 0.5
                
                # Volume (relaxed)
                (dataframe['volume_ratio'] > 1.2) &  # Was 2.5
                decent_volatility
            )
            
            aave_short = (
                # Momentum conditions (achievable)
                (dataframe['mom_10'] < 0) &
                (dataframe['roc_10'] < -0.1) &  # Was -0.5
                
                # RSI in reasonable zone
                (dataframe['rsi_14'] < 60) &  # Was 55
                (dataframe['rsi_14'] > 30) &  # Was 35
                
                # Trend alignment (relaxed)
                (dataframe['trend_score'] <= 0) &  # Was < -0.5
                
                # Volume (relaxed)
                (dataframe['volume_ratio'] > 1.2) &  # Was 2.5
                decent_volatility
            )
            
            dataframe.loc[aave_long, 'enter_long'] = 1
            dataframe.loc[aave_short, 'enter_short'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """BALANCED exit signals"""
        
        # Initialize exit columns
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        # Quick profit conditions
        quick_profit_long = (
            (dataframe['close'] > dataframe['close'].shift(1) * 1.008) |  # 0.8% quick profit
            (dataframe['bb_percent'] > 0.7) |  # Near BB middle/upper
            (dataframe['rsi_14'] > 70) |  # Overbought
            (dataframe['williams_r'] > -20)  # Williams %R exit
        )
        
        quick_profit_short = (
            (dataframe['close'] < dataframe['close'].shift(1) * 0.992) |  # 0.8% quick profit
            (dataframe['bb_percent'] < 0.3) |  # Near BB middle/lower
            (dataframe['rsi_14'] < 30) |  # Oversold
            (dataframe['williams_r'] < -80)  # Williams %R exit
        )
        
        # Trend reversal conditions
        trend_reversal_long = (
            (dataframe['macd'] < dataframe['macd_signal']) &
            (dataframe['ema_9'] < dataframe['ema_21'])
        )
        
        trend_reversal_short = (
            (dataframe['macd'] > dataframe['macd_signal']) &
            (dataframe['ema_9'] > dataframe['ema_21'])
        )
        
        # Apply exits
        dataframe.loc[quick_profit_long | trend_reversal_long, 'exit_long'] = 1
        dataframe.loc[quick_profit_short | trend_reversal_short, 'exit_short'] = 1
        
        return dataframe
