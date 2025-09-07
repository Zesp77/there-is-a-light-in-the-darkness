"""
Lighter Point Farming Scalp Trading Bot

Main entry point for the scalp trading system with Telegram notifications.
Supports multiple strategies, backtesting, and real-time signal generation.

Author: Lighter Point Farming Bot
Version: 1.0.0
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.bot_controller import ScalpBotController
from core.backtesting_engine import BacktestingEngine
from utils.logger import setup_logger

logger = setup_logger(__name__)

class LighterScalpBot:
    """
    Main application class for Lighter Point Farming Scalp Bot
    """

    def __init__(self):
        self.controller = None
        self.backtesting_engine = None

    async def run_live_trading(self, config_path: str, strategy_name: str):
        """
        Run live trading with Telegram notifications

        Args:
            config_path: Path to configuration file
            strategy_name: Name of strategy to use
        """
        try:
            logger.info(f"🚀 Starting Lighter Scalp Bot - Live Mode")
            logger.info(f"📊 Strategy: {strategy_name}")
            logger.info(f"⚙️ Config: {config_path}")

            # Initialize bot controller
            self.controller = ScalpBotController(config_path, strategy_name)

            # Start the bot
            await self.controller.start()

        except Exception as e:
            logger.error(f"❌ Error in live trading: {e}")
            raise

    async def run_backtest(self, config_path: str, strategy_name: str,
                          start_date: str, end_date: str, timeframe: str = "5m"):
        """
        Run backtesting for specified strategy

        Args:
            config_path: Path to configuration file
            strategy_name: Name of strategy to backtest
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            timeframe: Timeframe for backtesting
        """
        try:
            logger.info(f"🔍 Starting Backtest")
            logger.info(f"📊 Strategy: {strategy_name}")
            logger.info(f"📅 Period: {start_date} to {end_date}")
            logger.info(f"⏰ Timeframe: {timeframe}")

            # Initialize backtesting engine
            self.backtesting_engine = BacktestingEngine(config_path, strategy_name)

            # Run backtest (async)
            results = await self.backtesting_engine.run_backtest(
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )

            # Display results
            self._display_backtest_results(results)

        except Exception as e:
            logger.error(f"❌ Error in backtesting: {e}")
            raise

    def _display_backtest_results(self, results: dict):
        """Display formatted backtest results"""
        print("\n" + "="*60)
        print("🎯 BACKTEST RESULTS")
        print("="*60)
        print(f"💰 Total Trades: {results.get('total_trades', 0)}")
        print(f"✅ Winning Trades: {results.get('winning_trades', 0)}")
        print(f"❌ Losing Trades: {results.get('losing_trades', 0)}")
        print(f"📊 Win Rate: {results.get('win_rate', 0):.2f}%")
        print(f"💵 Total Profit: ${results.get('total_profit', 0):.2f}")
        print(f"📈 Total Profit %: {results.get('total_profit_percent', 0):.2f}%")
        print(f"📈 Average Profit per Trade: ${results.get('avg_profit', 0):.4f}")
        print(f"⚡ Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
        print(f"🔥 Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")

        print("\n📊 Strategy Performance by Pair:")
        for pair, stats in results.get('pair_stats', {}).items():
            print(f"  {pair}: {stats.get('trades', 0)} trades, "
                  f"${stats.get('profit', 0):.4f} profit")

        print("\n🎲 Signal Quality Distribution:")
        signal_dist = results.get('signal_distribution', {})
        print(f"  💎 GEM: {signal_dist.get('GEM', 0)} signals")
        print(f"  ✅ GOOD: {signal_dist.get('GOOD', 0)} signals")
        print(f"  ⚪ NEUTRAL: {signal_dist.get('NEUTRAL', 0)} signals")
        print("="*60)

def main():
    """Main function with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description='Lighter Point Farming Scalp Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run live trading
  python main.py live --config config/config_enhanced.json --strategy BalancedPointFarmingStrategy

  # Run backtest
  python main.py backtest --config config/config_backtest.json --strategy BalancedPointFarmingStrategy --start 2024-01-01 --end 2024-01-07

  # List strategies
  python main.py list-strategies
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Live trading command
    live_parser = subparsers.add_parser('live', help='Run live trading')
    live_parser.add_argument('--config', '-c', required=True,
                            help='Path to configuration file')
    live_parser.add_argument('--strategy', '-s', required=True,
                            help='Strategy name to use')

    # Backtesting command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')
    backtest_parser.add_argument('--config', '-c', required=True,
                                help='Path to configuration file')
    backtest_parser.add_argument('--strategy', '-s', required=True,
                                help='Strategy name to backtest')
    backtest_parser.add_argument('--start', required=True,
                                help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', required=True,
                                help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--timeframe', default='5m',
                                help='Timeframe for backtesting (default: 5m)')

    # List strategies command
    list_parser = subparsers.add_parser('list-strategies',
                                       help='List available strategies')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    bot = LighterScalpBot()

    try:
        if args.command == 'live':
            # Run live trading
            asyncio.run(bot.run_live_trading(args.config, args.strategy))

        elif args.command == 'backtest':
            # Run backtesting
            asyncio.run(bot.run_backtest(
                config_path=args.config,
                strategy_name=args.strategy,
                start_date=args.start,
                end_date=args.end,
                timeframe=args.timeframe
            ))

        elif args.command == 'list-strategies':
            # List available strategies
            from core.strategy_manager import StrategyManager
            manager = StrategyManager()
            strategies = manager.list_available_strategies()

            print("\n📊 Available Strategies:")
            print("="*40)
            for strategy in strategies:
                print(f"  • {strategy}")
            print("="*40)

    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
