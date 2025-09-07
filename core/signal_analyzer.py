"""
BALANCED Signal Analyzer - Achievable Quality Distribution

Fixed scoring system that produces realistic but achievable signal distribution
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
    BALANCED Signal Analyzer - Achievable quality distribution
    """

    def __init__(self):
        """Initialize with BALANCED thresholds for achievable signal distribution"""
        
        # BALANCED Quality thresholds - Achievable but meaningful
        self.quality_thresholds = {
            'GEM': {
                'min_score': 60,           # High but achievable bar for GEM signals
                'min_volume_ratio': 1.0,   # Good volume
                'min_volatility': 0.005,   # Decent volatility
                'min_momentum': 0.4        # Strong momentum required
            },
            'GOOD': {
                'min_score': 40,           # Medium bar for GOOD signals
                'min_volume_ratio': 0.7,   # Decent volume
                'min_volatility': 0.003,   # Some volatility
                'min_momentum': 0.25       # Moderate momentum
            },
            'NEUTRAL': {
                'min_score': 25,           # Lower bar for NEUTRAL
                'min_volume_ratio': 0.5,   # Minimal volume
                'min_volatility': 0.002,   # Low volatility ok
                'min_momentum': 0.1        # Low momentum ok
            }
        }

        # BALANCED scoring weights
        self.scoring_weights = {
            'momentum': 0.20,           # Important for scalping
            'volume': 0.20,             # Volume confirms moves
            'volatility': 0.15,         # Need movement for profit
            'trend_alignment': 0.25,    # Trend is very important
            'rsi_position': 0.20        # Entry timing
        }

        logger.info("⚖️ BALANCED Signal Analyzer initialized - Achievable quality distribution")

    def analyze_signal(self, signal: dict, market_data: pd.DataFrame,
                      config: dict) -> Optional[dict]:
        """Analyze and classify signals with BALANCED scoring"""
        try:
            # Calculate signal metrics
            metrics = self._calculate_signal_metrics(signal, market_data)

            # Calculate quality score with realistic expectations
            quality_score = self._calculate_quality_score(metrics)

            # Classify signal quality with ACHIEVABLE requirements
            quality = self._classify_signal_quality(quality_score, metrics)

            if quality is None:
                # Fallback to NEUTRAL if score is decent
                if quality_score >= 20:
                    quality = 'NEUTRAL'
                    logger.debug(f"🔄 Fallback NEUTRAL signal for {signal['pair']} (score: {quality_score:.1f})")
                else:
                    logger.debug(f"🚫 Signal rejected for {signal['pair']} (score: {quality_score:.1f})")
                    return None

            # Create enhanced signal
            enhanced_signal = {
                **signal,
                'quality': quality,
                'quality_score': quality_score,
                'metrics': metrics
            }
            
            # Calculate trade parameters
            trade_params = self._calculate_trade_parameters(enhanced_signal, metrics, config)
            enhanced_signal.update(trade_params)

            logger.info(f"✅ {quality} signal analyzed for {signal['pair']} "
                       f"(score: {quality_score:.1f})")

            return enhanced_signal

        except Exception as e:
            logger.error(f"❌ Error analyzing signal: {e}")
            return None

    def _calculate_signal_metrics(self, signal: dict, market_data: pd.DataFrame) -> dict:
        """Calculate metrics with BALANCED values"""
        try:
            latest_data = market_data.iloc[-1]
            recent_data = market_data.tail(20)

            metrics = {}

            # Volume analysis - BALANCED
            avg_volume = recent_data['volume'].mean()
            current_volume = latest_data['volume']
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                # Apply gentle minimum to avoid zero scores
                volume_ratio = max(volume_ratio, 0.3)
                metrics['volume_ratio'] = min(volume_ratio, 3.0)
            else:
                metrics['volume_ratio'] = 0.7  # Reasonable default

            # Volatility analysis - BALANCED
            high_low = recent_data['high'] - recent_data['low']
            high_close_prev = abs(recent_data['high'] - recent_data['close'].shift(1))
            low_close_prev = abs(recent_data['low'] - recent_data['close'].shift(1))
            true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            atr = true_range.mean()
            
            if latest_data['close'] > 0:
                volatility = atr / latest_data['close']
                # Apply gentle minimum to avoid zero scores
                volatility = max(volatility, 0.002)
                metrics['volatility'] = min(volatility, 0.08)
            else:
                metrics['volatility'] = 0.008

            # Momentum analysis - BALANCED
            momentum_score = 0.1  # Start with small base
            
            if len(recent_data) >= 6:
                price_change_5 = (latest_data['close'] - recent_data['close'].iloc[-6]) / recent_data['close'].iloc[-6]
                if signal['side'] == 'long':
                    momentum_score += max(0, price_change_5 * 8)
                else:
                    momentum_score += max(0, -price_change_5 * 8)
            
            if len(recent_data) >= 10:
                price_change_10 = (latest_data['close'] - recent_data['close'].iloc[-10]) / recent_data['close'].iloc[-10]
                if signal['side'] == 'long':
                    momentum_score += max(0, price_change_10 * 4)
                else:
                    momentum_score += max(0, -price_change_10 * 4)

            # Cap momentum score
            metrics['momentum'] = min(momentum_score, 1.0)

            # Trend alignment - BALANCED
            trend_score = 0.4  # Start with neutral
            
            if 'trend_1h' in signal.get('data', {}):
                trend_1h = signal['data']['trend_1h']
                if signal['side'] == 'long':
                    trend_score = 0.9 if trend_1h > 0 else 0.3
                else:
                    trend_score = 0.9 if trend_1h < 0 else 0.3

            # Check EMA alignment for additional points
            if 'ema_9' in signal.get('data', {}) and 'ema_21' in signal.get('data', {}):
                ema_9 = signal['data']['ema_9']
                ema_21 = signal['data']['ema_21']
                
                if signal['side'] == 'long' and ema_9 > ema_21:
                    trend_score = min(1.0, trend_score + 0.15)
                elif signal['side'] == 'short' and ema_9 < ema_21:
                    trend_score = min(1.0, trend_score + 0.15)

            metrics['trend_alignment'] = trend_score

            # RSI position - BALANCED
            rsi_score = 0.4  # Start with neutral
            
            if 'rsi_14' in signal.get('data', {}):
                rsi = signal['data']['rsi_14']
                if signal['side'] == 'long':
                    # Balanced RSI scoring for longs
                    if rsi <= 25:
                        rsi_score = 1.0
                    elif rsi <= 35:
                        rsi_score = 0.9
                    elif rsi <= 45:
                        rsi_score = 0.7
                    elif rsi <= 55:
                        rsi_score = 0.5
                    elif rsi <= 65:
                        rsi_score = 0.3
                    else:
                        rsi_score = 0.1
                else:
                    # Balanced RSI scoring for shorts
                    if rsi >= 75:
                        rsi_score = 1.0
                    elif rsi >= 65:
                        rsi_score = 0.9
                    elif rsi >= 55:
                        rsi_score = 0.7
                    elif rsi >= 45:
                        rsi_score = 0.5
                    elif rsi >= 35:
                        rsi_score = 0.3
                    else:
                        rsi_score = 0.1

            metrics['rsi_position'] = rsi_score

            # DEBUG LOGGING
            logger.debug(f"📊 Metrics for {signal['pair']}: "
                        f"vol={metrics['volume_ratio']:.2f}, "
                        f"volatility={metrics['volatility']:.4f}, "
                        f"momentum={metrics['momentum']:.2f}, "
                        f"trend={metrics['trend_alignment']:.2f}, "
                        f"rsi={metrics['rsi_position']:.2f}")

            return metrics

        except Exception as e:
            logger.error(f"❌ Error calculating signal metrics: {e}")
            # Return balanced default metrics
            return {
                'volume_ratio': 0.8,
                'volatility': 0.01,
                'momentum': 0.3,
                'trend_alignment': 0.5,
                'rsi_position': 0.4
            }

    def _calculate_quality_score(self, metrics: dict) -> float:
        """Calculate quality score with balanced expectations"""
        try:
            score = 0.0
            total_weight = 0.0
            
            for metric, weight in self.scoring_weights.items():
                if metric in metrics:
                    normalized_value = min(1.0, max(0.0, metrics[metric]))
                    score += normalized_value * weight * 100
                    total_weight += weight

            if total_weight > 0:
                # Add small base score to make scores more reasonable
                base_score = 10  # Small base score
                final_score = base_score + (score * 0.9)  # 90% of calculated score + base
                return min(100.0, max(0.0, final_score))
            
            return 10.0  # Minimum score

        except Exception as e:
            logger.error(f"❌ Error calculating quality score: {e}")
            return 10.0

    def _classify_signal_quality(self, quality_score: float, metrics: dict) -> Optional[str]:
        """Classify with BALANCED requirements"""
        try:
            logger.debug(f"📊 Classifying signal: score={quality_score:.1f}")
            
            # Check each quality level with BALANCED requirements
            for quality_level in ['GEM', 'GOOD', 'NEUTRAL']:
                thresholds = self.quality_thresholds[quality_level]

                # Check requirements (most must be met, not all)
                score_met = quality_score >= thresholds['min_score']
                volume_met = metrics.get('volume_ratio', 0) >= thresholds['min_volume_ratio']
                volatility_met = metrics.get('volatility', 0) >= thresholds['min_volatility']
                momentum_met = metrics.get('momentum', 0) >= thresholds['min_momentum']

                # Require score + at least 2 other conditions for GEM/GOOD
                # For NEUTRAL, just score is enough
                if quality_level == 'NEUTRAL':
                    conditions_met = sum([volume_met, volatility_met, momentum_met])
                    if score_met and conditions_met >= 1:  # Score + any 1 condition
                        logger.debug(f"✅ Signal classified as {quality_level}")
                        return quality_level
                else:
                    conditions_met = sum([volume_met, volatility_met, momentum_met])
                    if score_met and conditions_met >= 2:  # Score + any 2 conditions
                        logger.debug(f"✅ Signal classified as {quality_level}")
                        return quality_level

                logger.debug(f"❌ {quality_level} failed - score:{score_met}, vol:{volume_met}, "
                           f"volatility:{volatility_met}, momentum:{momentum_met}")

            # No classification met requirements
            return None

        except Exception as e:
            logger.error(f"❌ Error classifying signal quality: {e}")
            return None

    def _calculate_trade_parameters(self, signal: dict, metrics: dict, config: dict) -> dict:
        """Calculate trade parameters based on metrics"""
        try:
            entry_price = signal['price']
            pair = signal['pair']
            side = signal['side']

            # Base volatility for calculations
            volatility = metrics.get('volatility', 0.01)

            # Calculate stop loss based on volatility
            base_sl_percent = abs(config.get('stoploss', -0.008))
            volatility_multiplier = min(2.0, max(0.8, volatility * 40))
            sl_percent = base_sl_percent * volatility_multiplier

            if side == 'long':
                stop_loss = entry_price * (1 - sl_percent)
                tp1 = entry_price * (1 + sl_percent * 2.0)  # 2:1 R:R
                tp2 = entry_price * (1 + sl_percent * 3.0)  # 3:1 R:R
                tp3 = entry_price * (1 + sl_percent * 4.5)  # 4.5:1 R:R
            else:
                stop_loss = entry_price * (1 + sl_percent)
                tp1 = entry_price * (1 - sl_percent * 2.0)
                tp2 = entry_price * (1 - sl_percent * 3.0)
                tp3 = entry_price * (1 - sl_percent * 4.5)

            # Calculate leverage based on quality
            leverage = self._calculate_leverage(pair, signal.get('quality', 'NEUTRAL'), config)

            # Time limits based on quality
            time_limits = {
                'GEM': 45,     # Longer for high quality
                'GOOD': 60,    # Medium time
                'NEUTRAL': 90  # Longer for uncertain signals
            }

            return {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit_1': tp1,
                'take_profit_2': tp2,
                'take_profit_3': tp3,
                'recommended_leverage': leverage,
                'time_limit': time_limits.get(signal.get('quality', 'NEUTRAL'), 90),
                'sl_percent': sl_percent * 100,
                'risk_reward_ratio': 2.0
            }

        except Exception as e:
            logger.error(f"❌ Error calculating trade parameters: {e}")
            return {}

    def _calculate_leverage(self, pair: str, quality: str, config: dict) -> float:
        """Calculate leverage based on pair and quality"""
        try:
            base_leverage = {
                'ENA': 4.0,
                'LINK': 5.0,
                'AAVE': 4.5
            }

            pair_symbol = pair.split('/')[0]
            leverage = base_leverage.get(pair_symbol, 4.0)

            # Quality multipliers
            quality_multipliers = {
                'GEM': 1.4,      # Higher leverage for proven quality
                'GOOD': 1.2,     # Slight boost
                'NEUTRAL': 1.0   # Base leverage
            }

            leverage *= quality_multipliers.get(quality, 1.0)
            max_leverage = config.get('max_leverage', 10.0)
            
            return round(min(leverage, max_leverage), 1)

        except Exception as e:
            logger.error(f"❌ Error calculating leverage: {e}")
            return 4.0