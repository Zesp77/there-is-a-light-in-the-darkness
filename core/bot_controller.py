"""
Main bot controller for scalp trading system

Coordinates strategy execution, signal analysis, and Telegram notifications
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
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ScalpBotController:
    """
    Main controller for the scalp trading bot
    
    Manages strategy execution, signal generation, and notifications
    """
    
    def __init__(self, config_path: str, strategy_name: str):
        """
        Initialize bot controller
        
        Args:
            config_path: Path to configuration file
            strategy_name: Name of strategy to use
        """
        self.config_path = config_path
        self.strategy_name = strategy_name
        self.config = self._load_config()
        
        # Initialize components
        self.strategy_manager = StrategyManager()
        self.signal_analyzer = SignalAnalyzer()
        self.telegram_notifier = TelegramNotifier()
        
        # Runtime state
        self.is_running = False
        self.active_signals = {}
        self.trade_history = []
        
        # Performance tracking
        self.stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'total_profit': 0.0,
            'win_rate': 0.0
        }
        
        logger.info(f"🎯 Bot Controller initialized with strategy: {strategy_name}")
    
    def _load_config(self) -> dict:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Load Telegram config
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
        """Start the bot main loop"""
        try:
            logger.info("🚀 Starting Scalp Bot...")
            
            # Initialize strategy
            strategy = self.strategy_manager.load_strategy(self.strategy_name)
            if not strategy:
                raise Exception(f"Strategy {self.strategy_name} not found")
            
            # Initialize Telegram notifier
            if self.config['telegram'].get('enabled', False):
                await self.telegram_notifier.initialize(self.config['telegram'])
                await self.telegram_notifier.send_startup_message()
            
            self.is_running = True
            
            # Main trading loop
            while self.is_running:
                try:
                    await self._trading_cycle(strategy)
                    await asyncio.sleep(self.config.get('scan_interval', 30))
                    
                except Exception as e:
                    logger.error(f"❌ Error in trading cycle: {e}")
                    await asyncio.sleep(60)  # Wait longer on error
            
        except Exception as e:
            logger.error(f"❌ Critical error in bot: {e}")
            raise
    
    async def _trading_cycle(self, strategy):
        """Execute one trading cycle"""
        try:
            # Get pairs to analyze
            pairs = self.config['exchange']['pair_whitelist']
            
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
                        await self._process_signal(classified_signal)
            
            # Update active signals and check for exits
            await self._update_active_signals()
            
        except Exception as e:
            logger.error(f"❌ Error in trading cycle: {e}")
    
# Replace the _get_market_data method with:

async def _get_market_data(self, pair: str) -> Optional[pd.DataFrame]:
    """Get market data for a pair using the data provider"""
    try:
        from core.data_provider import get_market_data_provider
        
        data_provider = await get_market_data_provider(self.config)
        
        # Get OHLCV data for analysis
        market_data = await data_provider.get_ohlcv_data(
            pair=pair,
            timeframe=self.config.get('timeframe', '5m'),
            limit=200  # Get enough data for indicators
        )
        
        if market_data is None or market_data.empty:
            logger.warning(f"⚠️ No market data available for {pair}")
            return None
        
        logger.debug(f"📊 Retrieved {len(market_data)} candles for {pair}")
        return market_data
        
    except Exception as e:
        logger.error(f"❌ Error getting market data for {pair}: {e}")
        return None

    
    async def _generate_signals(self, strategy, pair: str, 
                               market_data: pd.DataFrame) -> List[dict]:
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
    
    async def _process_signal(self, signal: dict):
        """Process and potentially send a trading signal"""
        try:
            signal_id = f"{signal['pair']}_{signal['side']}_{signal['timestamp']}"
            
            # Check if we already processed this signal
            if signal_id in self.active_signals:
                return
            
            # Add to active signals
            self.active_signals[signal_id] = signal
            self.stats['signals_generated'] += 1
            
            # Send Telegram notification
            if self.config['telegram'].get('enabled', False):
                await self.telegram_notifier.send_trade_signal(signal)
            
            logger.info(f"🎯 {signal['quality']} signal generated for {signal['pair']} "
                       f"({signal['side']}): ${signal['entry_price']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error processing signal: {e}")
    
    async def _update_active_signals(self):
        """Update active signals and check for exits"""
        try:
            current_time = datetime.now()
            signals_to_remove = []
            
            for signal_id, signal in self.active_signals.items():
                # Check if signal has expired
                signal_age = current_time - signal['timestamp']
                if signal_age > timedelta(minutes=signal.get('time_limit', 60)):
                    signals_to_remove.append(signal_id)
                    
                    # Send expiry notification
                    if self.config['telegram'].get('enabled', False):
                        await self.telegram_notifier.send_signal_expired(signal)
            
            # Remove expired signals
            for signal_id in signals_to_remove:
                del self.active_signals[signal_id]
            
        except Exception as e:
            logger.error(f"❌ Error updating active signals: {e}")
    
    def stop(self):
        """Stop the bot"""
        logger.info("🛑 Stopping Scalp Bot...")
        self.is_running = False
