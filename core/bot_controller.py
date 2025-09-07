"""
Enhanced Bot Controller for Single Trade Execution

Coordinates strategy execution, signal analysis, and Telegram notifications
with single trade constraint optimization.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from .strategy_manager import StrategyManager
from .signal_analyzer import SignalAnalyzer
from .telegram_notifier import TelegramNotifier
from .data_provider import get_market_data_provider
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ScalpBotController:
    """
    Enhanced controller for single trade scalp bot
    Manages strategy execution, signal generation, and notifications
    """
    
    def __init__(self, config_path: str, strategy_name: str):
        """Initialize bot controller with single trade optimization"""
        self.config_path = config_path
        self.strategy_name = strategy_name
        self.config = self._load_config()
        
        # Initialize components
        self.strategy_manager = StrategyManager()
        self.signal_analyzer = SignalAnalyzer()
        self.telegram_notifier = TelegramNotifier()
        self.data_provider = None
        
        # Single trade state management
        self.is_running = False
        self.current_trade = None
        self.active_signals = {}
        self.trade_history = []
        
        # Performance tracking
        self.stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'total_profit': 0.0,
            'win_rate': 0.0
        }
        
        logger.info(f"🎯 Enhanced Bot Controller initialized - Single Trade Mode")
        logger.info(f"📊 Strategy: {strategy_name}")
    
    def _load_config(self) -> dict:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Load Telegram config if available
            telegram_config_path = 'config/telegram_config.json'
            try:
                with open(telegram_config_path, 'r') as f:
                    telegram_config = json.load(f)
                config['telegram'] = telegram_config
            except FileNotFoundError:
                logger.warning("⚠️ Telegram config not found, notifications disabled")
                config['telegram'] = {'enabled': False}
            
            return config
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            raise
    
    async def start(self):
        """Start the enhanced bot main loop"""
        try:
            logger.info("🚀 Starting Enhanced Scalp Bot - Single Trade Mode...")
            
            # Initialize data provider
            self.data_provider = await get_market_data_provider(self.config)
            
            # Initialize strategy
            strategy = self.strategy_manager.load_strategy(self.strategy_name)
            if not strategy:
                raise Exception(f"Strategy {self.strategy_name} not found")
            
            # Initialize Telegram notifier
            if self.config['telegram'].get('enabled', False):
                await self.telegram_notifier.initialize(self.config['telegram'])
                await self.telegram_notifier.send_startup_message()
            
            self.is_running = True
            
            # Enhanced trading loop
            while self.is_running:
                try:
                    await self._enhanced_trading_cycle(strategy)
                    await asyncio.sleep(self.config.get('scan_interval', 30))
                except Exception as e:
                    logger.error(f"❌ Error in trading cycle: {e}")
                    await asyncio.sleep(60)  # Wait longer on error
        
        except Exception as e:
            logger.error(f"❌ Critical error in bot: {e}")
            raise
        finally:
            if self.data_provider:
                await self.data_provider.close()
    
    async def _enhanced_trading_cycle(self, strategy):
        """Execute enhanced trading cycle with single trade constraint"""
        try:
            # Skip if we already have an active trade
            if self.current_trade is not None:
                await self._monitor_current_trade()
                return
            
            # Get pairs to analyze
            pairs = self.config['exchange']['pair_whitelist']
            
            # Analyze pairs and find the best signal
            best_signal = None
            best_quality_score = 0
            
            for pair in pairs:
                # Get market data
                market_data = await self._get_market_data(pair)
                if market_data is None:
                    continue
                
                # Generate signals using strategy
                signals = await self._generate_signals(strategy, pair, market_data)
                
                # Analyze and classify signals
                for signal in signals:
                    classified_signal = self.signal_analyzer.analyze_signal(
                        signal, market_data, self.config
                    )
                    
                    if classified_signal:
                        quality_score = classified_signal.get('quality_score', 0)
                        
                        # Only consider high-quality signals
                        if quality_score > 50 and quality_score > best_quality_score:
                            best_signal = classified_signal
                            best_quality_score = quality_score
            
            # Process the best signal if found
            if best_signal:
                await self._process_best_signal(best_signal)
            
            # Clean up expired signals
            await self._cleanup_expired_signals()
        
        except Exception as e:
            logger.error(f"❌ Error in enhanced trading cycle: {e}")
    
    async def _monitor_current_trade(self):
        """Monitor current active trade"""
        try:
            if not self.current_trade:
                return
            
            pair = self.current_trade['pair']
            
            # Get current market data
            market_data = await self._get_market_data(pair)
            if market_data is None:
                return
            
            current_price = market_data.iloc[-1]['close']
            
            # Calculate current P&L
            entry_price = self.current_trade['entry_price']
            side = self.current_trade['side']
            
            if side == 'long':
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_percent = ((entry_price - current_price) / entry_price) * 100
            
            # Check for exit conditions
            trade_duration = datetime.now() - self.current_trade['timestamp']
            
            should_exit = False
            exit_reason = ""
            
            # Time-based exit
            if trade_duration > timedelta(minutes=self.current_trade.get('time_limit', 60)):
                should_exit = True
                exit_reason = "Time limit reached"
            
            # Profit target
            elif pnl_percent >= 1.5:  # 1.5% profit target
                should_exit = True
                exit_reason = "Profit target hit"
            
            # Stop loss
            elif pnl_percent <= -0.8:  # 0.8% stop loss
                should_exit = True
                exit_reason = "Stop loss hit"
            
            if should_exit:
                await self._close_current_trade(exit_reason, pnl_percent)
            
            # Send progress update
            elif self.config['telegram'].get('enabled', False):
                if int(trade_duration.total_seconds()) % 300 == 0:  # Every 5 minutes
                    await self.telegram_notifier.send_trade_update(
                        self.current_trade, current_price, pnl_percent, 0, 
                        f"Trade progress: {pnl_percent:+.2f}%"
                    )
        
        except Exception as e:
            logger.error(f"❌ Error monitoring current trade: {e}")
    
    async def _close_current_trade(self, reason: str, pnl_percent: float):
        """Close current active trade"""
        try:
            if not self.current_trade:
                return
            
            self.current_trade['exit_reason'] = reason
            self.current_trade['pnl_percent'] = pnl_percent
            self.current_trade['exit_time'] = datetime.now()
            
            # Update statistics
            self.stats['trades_executed'] += 1
            self.stats['total_profit'] += pnl_percent
            
            if pnl_percent > 0:
                win_rate = (sum(1 for trade in self.trade_history if trade.get('pnl_percent', 0) > 0) + 1) / self.stats['trades_executed']
            else:
                win_rate = sum(1 for trade in self.trade_history if trade.get('pnl_percent', 0) > 0) / self.stats['trades_executed']
            
            self.stats['win_rate'] = win_rate * 100
            
            # Add to history
            self.trade_history.append(self.current_trade.copy())
            
            # Send notification
            if self.config['telegram'].get('enabled', False):
                await self.telegram_notifier.send_trade_update(
                    self.current_trade, 
                    self.current_trade['entry_price'],
                    pnl_percent, 
                    0,
                    f"Trade closed: {reason} | P&L: {pnl_percent:+.2f}%"
                )
            
            logger.info(f"📉 Trade closed: {self.current_trade['pair']} | "
                       f"P&L: {pnl_percent:+.2f}% | Reason: {reason}")
            
            # Clear current trade
            self.current_trade = None
        
        except Exception as e:
            logger.error(f"❌ Error closing current trade: {e}")
    
    async def _get_market_data(self, pair: str) -> Optional[pd.DataFrame]:
        """Get market data for a pair"""
        try:
            market_data = await self.data_provider.get_ohlcv_data(
                pair=pair,
                timeframe=self.config.get('timeframe', '5m'),
                limit=200
            )
            
            if market_data is None or market_data.empty:
                logger.warning(f"⚠️ No market data available for {pair}")
                return None
            
            logger.debug(f"📊 Retrieved {len(market_data)} candles for {pair}")
            return market_data
        
        except Exception as e:
            logger.error(f"❌ Error getting market data for {pair}: {e}")
            return None
    
    async def _generate_signals(self, strategy, pair: str, market_data: pd.DataFrame) -> List[dict]:
        """Generate trading signals using strategy"""
        try:
            # Apply strategy indicators
            analyzed_data = strategy.populate_indicators(
                market_data.copy(), {'pair': pair}
            )
            
            # Generate entry signals
            entry_data = strategy.populate_entry_trend(
                analyzed_data.copy(), {'pair': pair}
            )
            
            # Generate exit signals  
            exit_data = strategy.populate_exit_trend(
                entry_data.copy(), {'pair': pair}
            )
            
            # Extract signals from the data
            signals = []
            latest_row = exit_data.iloc[-1]
            
            # Check for long entry
            if latest_row.get('enter_long', 0) == 1:
                signals.append({
                    'pair': pair,
                    'side': 'long',
                    'type': 'entry',
                    'price': latest_row['close'],
                    'timestamp': datetime.now(),
                    'data': latest_row.to_dict()
                })
            
            # Check for short entry
            if latest_row.get('enter_short', 0) == 1:
                signals.append({
                    'pair': pair,
                    'side': 'short', 
                    'type': 'entry',
                    'price': latest_row['close'],
                    'timestamp': datetime.now(),
                    'data': latest_row.to_dict()
                })
            
            return signals
        
        except Exception as e:
            logger.error(f"❌ Error generating signals for {pair}: {e}")
            return []
    
    async def _process_best_signal(self, signal: dict):
        """Process the best quality signal found"""
        try:
            # Set as current trade
            self.current_trade = signal.copy()
            self.stats['signals_generated'] += 1
            
            # Send Telegram notification
            if self.config['telegram'].get('enabled', False):
                await self.telegram_notifier.send_trade_signal(signal)
            
            logger.info(f"🎯 Best {signal['quality']} signal selected for {signal['pair']} "
                       f"({signal['side']}): ${signal['entry_price']:.6f} | "
                       f"Score: {signal['quality_score']:.1f}")
        
        except Exception as e:
            logger.error(f"❌ Error processing best signal: {e}")
    
    async def _cleanup_expired_signals(self):
        """Clean up expired signals"""
        try:
            current_time = datetime.now()
            signals_to_remove = []
            
            for signal_id, signal in self.active_signals.items():
                signal_age = current_time - signal['timestamp']
                if signal_age > timedelta(minutes=60):
                    signals_to_remove.append(signal_id)
            
            for signal_id in signals_to_remove:
                del self.active_signals[signal_id]
        
        except Exception as e:
            logger.error(f"❌ Error cleaning up expired signals: {e}")
    
    def stop(self):
        """Stop the bot"""
        logger.info("🛑 Stopping Enhanced Scalp Bot...")
        self.is_running = False
