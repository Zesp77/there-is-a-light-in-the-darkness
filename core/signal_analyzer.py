"""
Signal analysis and classification system

Analyzes trading signals and classifies them as GEM, GOOD, or NEUTRAL
based on multiple factors including volatility, volume, momentum, and risk/reward
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from utils.logger import setup_logger

logger = setup_logger(__name__)

class SignalAnalyzer:
    """
    Analyzes and classifies trading signals based on quality metrics
    """
    
    def __init__(self):
        """Initialize signal analyzer with quality thresholds"""
        
        # Quality thresholds for classification
        self.quality_thresholds = {
            'GEM': {
                'min_score': 80,
                'min_volume_ratio': 2.0,
                'min_volatility': 0.02,
                'max_risk_reward': 0.3,  # SL/TP ratio
                'min_momentum': 0.7
            },
            'GOOD': {
                'min_score': 60,
                'min_volume_ratio': 1.5,
                'min_volatility': 0.015,
                'max_risk_reward': 0.4,
                'min_momentum': 0.5
            },
            'NEUTRAL': {
                'min_score': 40,
                'min_volume_ratio': 1.0,
                'min_volatility': 0.01,
                'max_risk_reward': 0.6,
                'min_momentum': 0.3
            }
        }
        
        # Scoring weights
        self.scoring_weights = {
            'momentum': 0.25,
            'volume': 0.20,
            'volatility': 0.15,
            'trend_alignment': 0.15,
            'rsi_position': 0.10,
            'risk_reward': 0.15
        }
        
        logger.info("🎯 Signal Analyzer initialized")
    
    def analyze_signal(self, signal: dict, market_data: pd.DataFrame, 
                      config: dict) -> Optional[dict]:
        """
        Analyze and classify a trading signal
        
        Args:
            signal: Raw signal from strategy
            market_data: Market data for analysis
            config: Bot configuration
            
        Returns:
            Enhanced signal with quality classification or None if signal is rejected
        """
        try:
            # Calculate signal metrics
            metrics = self._calculate_signal_metrics(signal, market_data)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(metrics)
            
            # Classify signal quality
            quality = self._classify_signal_quality(quality_score, metrics)
            
            if quality is None:
                logger.debug(f"🚫 Signal rejected for {signal['pair']} (score: {quality_score:.1f})")
                return None
            
            # Calculate trade parameters
            trade_params = self._calculate_trade_parameters(signal, metrics, config)
            
            # Enhance signal with analysis results
            enhanced_signal = {
                **signal,
                'quality': quality,
                'quality_score': quality_score,
                'metrics': metrics,
                **trade_params
            }
            
            logger.info(f"✅ {quality} signal analyzed for {signal['pair']} "
                       f"(score: {quality_score:.1f})")
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"❌ Error analyzing signal: {e}")
            return None
    
    def _calculate_signal_metrics(self, signal: dict, 
                                 market_data: pd.DataFrame) -> dict:
        """Calculate various metrics for signal quality assessment"""
        try:
            latest_data = market_data.iloc[-1]
            recent_data = market_data.tail(20)  # Last 20 candles
            
            metrics = {}
            
            # Volume analysis
            avg_volume = recent_data['volume'].mean()
            current_volume = latest_data['volume']
            metrics['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volatility analysis (ATR-based)
            high_low = recent_data['high'] - recent_data['low']
            high_close_prev = abs(recent_data['high'] - recent_data['close'].shift(1))
            low_close_prev = abs(recent_data['low'] - recent_data['close'].shift(1))
            
            true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            atr = true_range.mean()
            metrics['volatility'] = atr / latest_data['close'] if latest_data['close'] > 0 else 0
            
            # Momentum analysis
            price_change_5 = (latest_data['close'] - recent_data['close'].iloc[-6]) / recent_data['close'].iloc[-6]
            price_change_20 = (latest_data['close'] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            
            momentum_score = 0
            if signal['side'] == 'long':
                momentum_score = max(0, min(1, (price_change_5 * 2 + price_change_20) * 10 + 0.5))
            else:
                momentum_score = max(0, min(1, (-price_change_5 * 2 - price_change_20) * 10 + 0.5))
            
            metrics['momentum'] = momentum_score
            
            # Trend alignment
            if 'trend_1h' in signal.get('data', {}):
                trend_1h = signal['data']['trend_1h']
                if signal['side'] == 'long' and trend_1h > 0:
                    metrics['trend_alignment'] = 1.0
                elif signal['side'] == 'short' and trend_1h < 0:
                    metrics['trend_alignment'] = 1.0
                else:
                    metrics['trend_alignment'] = 0.3  # Counter-trend
            else:
                metrics['trend_alignment'] = 0.5  # Neutral
            
            # RSI position analysis
            if 'rsi_14' in signal.get('data', {}):
                rsi = signal['data']['rsi_14']
                if signal['side'] == 'long':
                    # For longs, prefer RSI between 25-45 (oversold but recovering)
                    if 25 <= rsi <= 45:
                        metrics['rsi_position'] = 1.0
                    elif rsi < 25:
                        metrics['rsi_position'] = 0.7
                    else:
                        metrics['rsi_position'] = 0.3
                else:
                    # For shorts, prefer RSI between 55-75 (overbought but declining)
                    if 55 <= rsi <= 75:
                        metrics['rsi_position'] = 1.0
                    elif rsi > 75:
                        metrics['rsi_position'] = 0.7
                    else:
                        metrics['rsi_position'] = 0.3
            else:
                metrics['rsi_position'] = 0.5
            
            # Market conditions
            metrics['market_hour'] = self._get_market_hour_score()
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating signal metrics: {e}")
            return {}
    
    def _calculate_quality_score(self, metrics: dict) -> float:
        """Calculate overall quality score based on metrics"""
        try:
            score = 0.0
            
            for metric, weight in self.scoring_weights.items():
                if metric in metrics:
                    # Normalize metric to 0-1 scale and apply weight
                    normalized_value = min(1.0, max(0.0, metrics[metric]))
                    score += normalized_value * weight * 100
            
            return min(100.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"❌ Error calculating quality score: {e}")
            return 0.0
    
    def _classify_signal_quality(self, quality_score: float, metrics: dict) -> Optional[str]:
        """Classify signal quality based on score and metrics"""
        try:
            # Check each quality level from highest to lowest
            for quality_level in ['GEM', 'GOOD', 'NEUTRAL']:
                thresholds = self.quality_thresholds[quality_level]
                
                # Check if signal meets all requirements for this quality level
                meets_requirements = (
                    quality_score >= thresholds['min_score'] and
                    metrics.get('volume_ratio', 0) >= thresholds['min_volume_ratio'] and
                    metrics.get('volatility', 0) >= thresholds['min_volatility'] and
                    metrics.get('momentum', 0) >= thresholds['min_momentum']
                )
                
                if meets_requirements:
                    return quality_level
            
            return None  # Signal doesn't meet minimum requirements
            
        except Exception as e:
            logger.error(f"❌ Error classifying signal quality: {e}")
            return None
    
    def _calculate_trade_parameters(self, signal: dict, metrics: dict, 
                                   config: dict) -> dict:
        """Calculate trade parameters (TP, SL, leverage, etc.)"""
        try:
            entry_price = signal['price']
            pair = signal['pair']
            side = signal['side']
            
            # Base volatility for calculations
            volatility = metrics.get('volatility', 0.02)
            
            # Calculate stop loss (adaptive based on volatility)
            base_sl_percent = config.get('stoploss', -0.02)  # Default 2%
            volatility_multiplier = min(2.0, max(0.5, volatility * 50))  # Scale with volatility
            sl_percent = abs(base_sl_percent) * volatility_multiplier
            
            if side == 'long':
                stop_loss = entry_price * (1 - sl_percent)
                # Take profit targets
                tp1 = entry_price * (1 + sl_percent * 1.5)  # 1.5:1 R:R
                tp2 = entry_price * (1 + sl_percent * 2.5)  # 2.5:1 R:R
                tp3 = entry_price * (1 + sl_percent * 4.0)  # 4:1 R:R
            else:
                stop_loss = entry_price * (1 + sl_percent)
                tp1 = entry_price * (1 - sl_percent * 1.5)
                tp2 = entry_price * (1 - sl_percent * 2.5)
                tp3 = entry_price * (1 - sl_percent * 4.0)
            
            # Calculate recommended leverage based on pair and quality
            leverage = self._calculate_leverage(pair, signal['quality'], config)
            
            # Time limits based on signal quality
            time_limits = {
                'GEM': 30,     # 30 minutes for GEM signals
                'GOOD': 45,    # 45 minutes for GOOD signals
                'NEUTRAL': 60  # 60 minutes for NEUTRAL signals
            }
            
            return {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit_1': tp1,
                'take_profit_2': tp2,
                'take_profit_3': tp3,
                'recommended_leverage': leverage,
                'time_limit': time_limits.get(signal['quality'], 60),
                'sl_percent': sl_percent * 100,  # Convert to percentage
                'risk_reward_ratio': 1.5  # Base R:R ratio
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating trade parameters: {e}")
            return {}
    
    def _calculate_leverage(self, pair: str, quality: str, config: dict) -> float:
        """Calculate recommended leverage based on pair and signal quality"""
        try:
            # Base leverage from config or defaults
            base_leverage = {
                'ENA': 4.0,   # Higher volatility
                'LINK': 5.0,  # Medium volatility
                'AAVE': 3.0   # Lower volatility for breakouts
            }
            
            # Get base leverage for pair
            pair_symbol = pair.split('/')[0]
            leverage = base_leverage.get(pair_symbol, 4.0)
            
            # Adjust based on signal quality
            quality_multipliers = {
                'GEM': 1.2,     # Increase leverage for high-quality signals
                'GOOD': 1.0,    # Standard leverage
                'NEUTRAL': 0.8  # Reduce leverage for uncertain signals
            }
            
            leverage *= quality_multipliers.get(quality, 1.0)
            
            # Respect maximum leverage from config
            max_leverage = config.get('max_leverage', 10.0)
            leverage = min(leverage, max_leverage)
            
            return round(leverage, 1)
            
        except Exception as e:
            logger.error(f"❌ Error calculating leverage: {e}")
            return 3.0  # Safe default
    
    def _get_market_hour_score(self) -> float:
        """Get market activity score based on current time"""
        try:
            current_hour = datetime.now().hour
            
            # Peak trading hours (UTC): 8-10 (EU open), 13-15 (US open), 21-23 (Asia)
            peak_hours = [8, 9, 10, 13, 14, 15, 21, 22, 23]
            
            if current_hour in peak_hours:
                return 1.0
            elif current_hour in [7, 11, 12, 16, 20, 0]:  # Moderate activity
                return 0.7
            else:  # Low activity hours
                return 0.4
                
        except Exception as e:
            logger.error(f"❌ Error getting market hour score: {e}")
            return 0.5
