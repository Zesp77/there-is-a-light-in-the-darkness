"""
Backtesting Engine for Lighter Point Farming Bot

Comprehensive backtesting system for strategy validation and optimization.
Supports historical data analysis, performance metrics, and detailed reporting.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import os
import json  # ADD THIS LINE
import asyncio
from pathlib import Path

from .strategy_manager import StrategyManager
from .data_provider import MarketDataProvider, get_market_data_provider
from .signal_analyzer import SignalAnalyzer
from utils.logger import setup_logger

logger = setup_logger(__name__)

class BacktestingEngine:
    """
    Comprehensive backtesting engine for strategy validation
    """
    
    def __init__(self, config_path: str, strategy_name: str):
        """
        Initialize backtesting engine
        
        Args:
            config_path: Path to configuration file
            strategy_name: Name of strategy to backtest
        """
        self.config_path = config_path
        self.strategy_name = strategy_name
        self.config = self._load_config()
        
        # Initialize components
        self.strategy_manager = StrategyManager()
        self.signal_analyzer = SignalAnalyzer()
        self.data_provider = None
        
        # Backtesting state
        self.results = {}
        self.trades = []
        self.signals = []
        self.portfolio_value = []
        
        # Performance metrics
        self.initial_balance = 10000.0  # Default starting balance
        self.current_balance = self.initial_balance
        self.max_drawdown = 0.0
        self.peak_value = self.initial_balance
        
        logger.info(f"🔍 Backtesting Engine initialized for {strategy_name}")
    
    def _load_config(self) -> dict:
        """Load configuration from file"""
        try:
            # Try the provided path first
            config_path = self.config_path
            
            # If it doesn't exist, try looking in current directory or with config/ prefix
            if not os.path.exists(config_path):
                config_filename = os.path.basename(config_path)
                # Try with config/ prefix
                alt_path = os.path.join('config', config_filename) if not config_path.startswith('config/') else config_path
                if os.path.exists(alt_path):
                    config_path = alt_path
                elif os.path.exists(config_filename):
                    config_path = config_filename
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"✅ Successfully loaded config from: {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            return {}



    
    async def run_backtest(self, start_date: str, end_date: str, 
                          timeframe: str = "5m") -> dict:
        """
        Run comprehensive backtest
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Data timeframe
            
        Returns:
            Backtesting results dictionary
        """
        try:
            logger.info(f"🚀 Starting backtest: {self.strategy_name}")
            logger.info(f"📅 Period: {start_date} to {end_date}")
            logger.info(f"⏰ Timeframe: {timeframe}")
            
            # Initialize data provider
            self.data_provider = await get_market_data_provider(self.config)
            
            # Load strategy
            strategy = self.strategy_manager.load_strategy(self.strategy_name)
            if not strategy:
                raise Exception(f"Could not load strategy: {self.strategy_name}")
            
            # Get pairs to test
            pairs = self.config.get('exchange', {}).get('pair_whitelist', [])
            if not pairs:
                raise Exception("No pairs configured for backtesting")
            
            # Initialize backtesting state
            self._initialize_backtest(start_date, end_date)
            
            # Run backtest for each pair
            for pair in pairs:
                await self._backtest_pair(strategy, pair, start_date, end_date, timeframe)
            
            # Calculate final results
            results = self._calculate_results()
            
            # Generate detailed report
            self._generate_report(results)
            
            logger.info(f"✅ Backtest completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error running backtest: {e}")
            return {'error': str(e)}
    
    def _initialize_backtest(self, start_date: str, end_date: str):
        """Initialize backtesting state"""
        try:
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
            self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Reset state
            self.trades = []
            self.signals = []
            self.portfolio_value = []
            
            # Initialize balance
            self.initial_balance = self.config.get('dry_run_wallet', 10000.0)
            self.current_balance = self.initial_balance
            self.max_drawdown = 0.0
            self.peak_value = self.initial_balance
            
            # Portfolio tracking
            self.portfolio_value.append({
                'timestamp': self.start_date,
                'balance': self.current_balance,
                'drawdown': 0.0
            })
            
            logger.debug(f"💰 Initialized with balance: ${self.initial_balance:,.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error initializing backtest: {e}")
            raise
    
    async def _backtest_pair(self, strategy, pair: str, start_date: str, 
                           end_date: str, timeframe: str):
        """
        Backtest strategy for a specific pair
        
        Args:
            strategy: Strategy instance
            pair: Trading pair
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
        """
        try:
            logger.info(f"📊 Backtesting {pair}...")
            
            # Get historical data
            historical_data = await self.data_provider.get_historical_data_for_backtest(
                pair=pair,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if historical_data is None or historical_data.empty:
                logger.warning(f"⚠️ No historical data for {pair}")
                return
            
            logger.info(f"📈 Retrieved {len(historical_data)} candles for {pair}")
            
            # Apply strategy indicators
            analyzed_data = strategy.populate_indicators(
                historical_data.copy(), {'pair': pair}
            )
            
            # Generate entry signals
            entry_data = strategy.populate_entry_trend(
                analyzed_data.copy(), {'pair': pair}
            )
            
            # Generate exit signals
            exit_data = strategy.populate_exit_trend(
                entry_data.copy(), {'pair': pair}
            )
            
            # Process signals chronologically
            await self._process_signals(strategy, pair, exit_data)
            
        except Exception as e:
            logger.error(f"❌ Error backtesting {pair}: {e}")
    
    async def _process_signals(self, strategy, pair: str, data: pd.DataFrame):
        """
        Process signals chronologically and simulate trading
        
        Args:
            strategy: Strategy instance
            pair: Trading pair
            data: DataFrame with signals and indicators
        """
        try:
            open_trades = {}  # Track open trades
            
            for idx, row in data.iterrows():
                try:
                    current_time = row['timestamp']
                    
                    # Process open trades first (check for exits)
                    trades_to_close = []
                    for trade_id, trade in open_trades.items():
                        exit_result = self._check_trade_exit(
                            strategy, trade, row, current_time
                        )
                        if exit_result:
                            trades_to_close.append(trade_id)
                            await self._close_trade(trade, exit_result, current_time)
                    
                    # Remove closed trades
                    for trade_id in trades_to_close:
                        del open_trades[trade_id]
                    
                    # Check for new entry signals
                    entry_signal = self._check_entry_signal(strategy, pair, row)
                    if entry_signal:
                        # Analyze signal quality
                        analyzed_signal = self.signal_analyzer.analyze_signal(
                            entry_signal, data.loc[:idx], self.config
                        )
                        
                        if analyzed_signal and self._should_take_trade(analyzed_signal):
                            trade = await self._open_trade(analyzed_signal, current_time)
                            if trade:
                                open_trades[trade['id']] = trade
                    
                    # Update portfolio value
                    self._update_portfolio_value(current_time, open_trades)
                    
                except Exception as e:
                    logger.debug(f"⚠️ Error processing signal at {row.get('timestamp')}: {e}")
                    continue
            
            # Close any remaining open trades at the end
            for trade in open_trades.values():
                await self._close_trade(trade, {'reason': 'backtest_end'}, self.end_date)
            
        except Exception as e:
            logger.error(f"❌ Error processing signals: {e}")
    
    def _check_entry_signal(self, strategy, pair: str, row: pd.Series) -> Optional[dict]:
        """Check for entry signals in the current row"""
        try:
            signal = None
            
            # Check for long signal
            if row.get('enter_long', 0) == 1:
                signal = {
                    'pair': pair,
                    'side': 'long',
                    'type': 'entry',
                    'price': row['close'],
                    'timestamp': row['timestamp'],
                    'data': row.to_dict()
                }
            
            # Check for short signal
            elif row.get('enter_short', 0) == 1:
                signal = {
                    'pair': pair,
                    'side': 'short', 
                    'type': 'entry',
                    'price': row['close'],
                    'timestamp': row['timestamp'],
                    'data': row.to_dict()
                }
            
            return signal
            
        except Exception as e:
            logger.debug(f"⚠️ Error checking entry signal: {e}")
            return None
    
    def _check_trade_exit(self, strategy, trade: dict, row: pd.Series, 
                         current_time: datetime) -> Optional[dict]:
        """Check if trade should be exited"""
        try:
            # Check exit signals
            if trade['side'] == 'long' and row.get('exit_long', 0) == 1:
                return {
                    'reason': 'exit_signal',
                    'price': row['close']
                }
            elif trade['side'] == 'short' and row.get('exit_short', 0) == 1:
                return {
                    'reason': 'exit_signal',
                    'price': row['close']
                }
            
            # Check stop loss
            current_price = row['close']
            if self._check_stop_loss(trade, current_price):
                return {
                    'reason': 'stop_loss',
                    'price': current_price
                }
            
            # Check take profit
            if self._check_take_profit(trade, current_price):
                return {
                    'reason': 'take_profit',
                    'price': current_price
                }
            
            # Check time limit
            trade_duration = current_time - trade['entry_time']
            if trade_duration > timedelta(hours=1):  # 1 hour time limit
                return {
                    'reason': 'time_limit',
                    'price': current_price
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"⚠️ Error checking trade exit: {e}")
            return None
    
    def _check_stop_loss(self, trade: dict, current_price: float) -> bool:
        """Check if stop loss should trigger"""
        try:
            stop_loss = trade.get('stop_loss')
            if not stop_loss:
                return False
            
            if trade['side'] == 'long':
                return current_price <= stop_loss
            else:  # short
                return current_price >= stop_loss
                
        except Exception as e:
            logger.debug(f"⚠️ Error checking stop loss: {e}")
            return False
    
    def _check_take_profit(self, trade: dict, current_price: float) -> bool:
        """Check if take profit should trigger"""
        try:
            # Check multiple take profit levels
            for i in range(1, 4):  # TP1, TP2, TP3
                tp_key = f'take_profit_{i}'
                if tp_key in trade:
                    tp_level = trade[tp_key]
                    
                    if trade['side'] == 'long':
                        if current_price >= tp_level:
                            return True
                    else:  # short
                        if current_price <= tp_level:
                            return True
            
            return False
            
        except Exception as e:
            logger.debug(f"⚠️ Error checking take profit: {e}")
            return False
    
    def _should_take_trade(self, signal: dict) -> bool:
        """Determine if trade should be taken based on signal quality"""
        try:
            # Only take GOOD or GEM signals in backtesting
            quality = signal.get('quality', 'NEUTRAL')
            return quality in ['GOOD', 'GEM']
            
        except Exception as e:
            logger.debug(f"⚠️ Error checking trade eligibility: {e}")
            return False
    
    async def _open_trade(self, signal: dict, entry_time: datetime) -> Optional[dict]:
        """Open a new trade"""
        try:
            trade_id = f"trade_{len(self.trades) + 1}"
            
            # Calculate position size (simplified)
            risk_amount = self.current_balance * 0.02  # 2% risk per trade
            
            trade = {
                'id': trade_id,
                'pair': signal['pair'],
                'side': signal['side'],
                'entry_price': signal['entry_price'],
                'stop_loss': signal.get('stop_loss'),
                'take_profit_1': signal.get('take_profit_1'),
                'take_profit_2': signal.get('take_profit_2'),
                'take_profit_3': signal.get('take_profit_3'),
                'leverage': signal.get('recommended_leverage', 3.0),
                'entry_time': entry_time,
                'risk_amount': risk_amount,
                'quality': signal.get('quality'),
                'status': 'open'
            }
            
            self.signals.append(signal)
            
            logger.debug(f"📈 Opened {trade['side']} trade for {trade['pair']} at ${trade['entry_price']:.6f}")
            
            return trade
            
        except Exception as e:
            logger.error(f"❌ Error opening trade: {e}")
            return None
    
    async def _close_trade(self, trade: dict, exit_result: dict, exit_time: datetime):
        """Close an existing trade"""
        try:
            exit_price = exit_result['price']
            exit_reason = exit_result['reason']
            
            # Calculate profit/loss
            entry_price = trade['entry_price']
            leverage = trade['leverage']
            
            if trade['side'] == 'long':
                price_change = (exit_price - entry_price) / entry_price
            else:  # short
                price_change = (entry_price - exit_price) / entry_price
            
            # Apply leverage
            pnl_percent = price_change * leverage
            pnl_amount = trade['risk_amount'] * pnl_percent
            
            # Update balance
            self.current_balance += pnl_amount
            
            # Complete trade record
            trade.update({
                'exit_price': exit_price,
                'exit_time': exit_time,
                'exit_reason': exit_reason,
                'pnl_percent': pnl_percent * 100,
                'pnl_amount': pnl_amount,
                'duration': exit_time - trade['entry_time'],
                'status': 'closed'
            })
            
            self.trades.append(trade)
            
            logger.debug(f"📉 Closed {trade['side']} trade for {trade['pair']}: "
                        f"{trade['pnl_percent']:+.2f}% (${trade['pnl_amount']:+.2f})")
            
        except Exception as e:
            logger.error(f"❌ Error closing trade: {e}")
    
    def _update_portfolio_value(self, timestamp: datetime, open_trades: dict):
        """Update portfolio value tracking"""
        try:
            # Calculate unrealized PnL from open trades
            unrealized_pnl = 0.0
            # Note: In a real backtest, you'd calculate unrealized PnL here
            
            total_value = self.current_balance + unrealized_pnl
            
            # Update peak value and drawdown
            if total_value > self.peak_value:
                self.peak_value = total_value
            
            current_drawdown = (self.peak_value - total_value) / self.peak_value
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
            
            # Store portfolio value
            self.portfolio_value.append({
                'timestamp': timestamp,
                'balance': total_value,
                'drawdown': current_drawdown * 100
            })
            
        except Exception as e:
            logger.debug(f"⚠️ Error updating portfolio value: {e}")
    
    def _calculate_results(self) -> dict:
        """Calculate comprehensive backtest results"""
        try:
            if not self.trades:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_profit': 0,
                    'total_profit_percent': 0,
                    'avg_profit': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0,
                    'profit_factor': 0,
                    'pair_stats': {},
                    'signal_distribution': {}
                }
            
            # Basic statistics
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t['pnl_amount'] > 0])
            losing_trades = total_trades - winning_trades
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Profit statistics
            total_profit = sum(t['pnl_amount'] for t in self.trades)
            total_profit_percent = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
            avg_profit = total_profit / total_trades if total_trades > 0 else 0
            
            # Risk metrics
            profits = [t['pnl_amount'] for t in self.trades if t['pnl_amount'] > 0]
            losses = [abs(t['pnl_amount']) for t in self.trades if t['pnl_amount'] < 0]
            
            gross_profit = sum(profits) if profits else 0
            gross_loss = sum(losses) if losses else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Sharpe ratio (simplified)
            returns = [t['pnl_percent'] for t in self.trades]
            if returns and len(returns) > 1:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Per-pair statistics
            pair_stats = {}
            for trade in self.trades:
                pair = trade['pair']
                if pair not in pair_stats:
                    pair_stats[pair] = {'trades': 0, 'profit': 0}
                pair_stats[pair]['trades'] += 1
                pair_stats[pair]['profit'] += trade['pnl_amount']
            
            # Signal quality distribution
            signal_dist = {}
            for signal in self.signals:
                quality = signal.get('quality', 'UNKNOWN')
                signal_dist[quality] = signal_dist.get(quality, 0) + 1
            
            results = {
                'strategy_name': self.strategy_name,
                'start_date': self.start_date.strftime('%Y-%m-%d'),
                'end_date': self.end_date.strftime('%Y-%m-%d'),
                'initial_balance': self.initial_balance,
                'final_balance': self.current_balance,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'total_profit_percent': total_profit_percent,
                'avg_profit': avg_profit,
                'max_drawdown': self.max_drawdown * 100,
                'sharpe_ratio': sharpe_ratio,
                'profit_factor': profit_factor,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'pair_stats': pair_stats,
                'signal_distribution': signal_dist,
                'trades': self.trades,
                'portfolio_value': self.portfolio_value
            }
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error calculating results: {e}")
            return {'error': str(e)}
    
    def _generate_report(self, results: dict):
        """Generate detailed backtest report"""
        try:
            # Create reports directory
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{self.strategy_name}_{timestamp}.json"
            filepath = reports_dir / filename
            
            # Save detailed results
            import json
            with open(filepath, 'w') as f:
                # Convert datetime objects to strings for JSON serialization
                serializable_results = self._make_json_serializable(results)
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"📊 Backtest report saved: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Error generating report: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON serializable format"""
        try:
            if isinstance(obj, dict):
                return {k: self._make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self._make_json_serializable(item) for item in obj]
            elif isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            elif isinstance(obj, timedelta):
                return str(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif pd.isna(obj):
                return None
            else:
                return obj
        except Exception:
            return str(obj)
