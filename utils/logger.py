"""
Logging Configuration for Lighter Point Farming Bot

Provides centralized logging setup with proper formatting, file rotation,
and different log levels for development and production environments.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
import colorama
from colorama import Fore, Back, Style

# Initialize colorama for cross-platform colored output
colorama.init()

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    # Color mapping for different log levels
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT
    }
    
    def format(self, record):
        # Save the original format
        original_format = self._style._fmt
        
        # Get color for the log level
        color = self.COLORS.get(record.levelname, Fore.WHITE)
        
        # Add color to the log level name
        colored_levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        
        # Create colored format
        colored_format = original_format.replace(
            '%(levelname)s', colored_levelname
        )
        
        # Apply the colored format
        self._style._fmt = colored_format
        
        # Format the record
        formatted = super().format(record)
        
        # Restore original format
        self._style._fmt = original_format
        
        return formatted

def setup_logger(name: str = __name__, 
                level: str = "INFO",
                log_to_file: bool = True,
                log_to_console: bool = True) -> logging.Logger:
    """
    Set up logger with file and console handlers
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Set log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Define log formats
    detailed_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    )
    
    simple_format = (
        "%(asctime)s | %(levelname)-8s | %(message)s"
    )
    
    # Console handler with colors
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        console_formatter = ColoredFormatter(
            fmt=simple_format,
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_to_file:
        # Main log file
        log_file = logs_dir / "scalp_bot.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        
        file_formatter = logging.Formatter(
            fmt=detailed_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Separate error log file
        error_log_file = logs_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger

def setup_trading_logger() -> logging.Logger:
    """Set up specialized logger for trading activities"""
    
    logger = setup_logger("TRADING", level="INFO")
    
    # Add specialized trading log file
    logs_dir = Path("logs")
    trading_log_file = logs_dir / "trading.log"
    
    trading_handler = logging.handlers.RotatingFileHandler(
        trading_log_file,
        maxBytes=20 * 1024 * 1024,  # 20MB
        backupCount=10,
        encoding='utf-8'
    )
    
    trading_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | TRADE | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    trading_handler.setFormatter(trading_formatter)
    logger.addHandler(trading_handler)
    
    return logger

def setup_performance_logger() -> logging.Logger:
    """Set up specialized logger for performance metrics"""
    
    logger = setup_logger("PERFORMANCE", level="INFO")
    
    # Add performance log file
    logs_dir = Path("logs")
    perf_log_file = logs_dir / "performance.log"
    
    perf_handler = logging.handlers.RotatingFileHandler(
        perf_log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    
    perf_formatter = logging.Formatter(
        fmt="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    perf_handler.setFormatter(perf_formatter)
    logger.addHandler(perf_handler)
    
    return logger

def log_trade_signal(pair: str, side: str, price: float, quality: str, **kwargs):
    """
    Log a trade signal with standardized format
    
    Args:
        pair: Trading pair
        side: Trade side (long/short)
        price: Entry price
        quality: Signal quality (GEM/GOOD/NEUTRAL)
        **kwargs: Additional signal data
    """
    
    trading_logger = logging.getLogger("TRADING")
    
    message = (
        f"SIGNAL | {quality} | {pair} | {side.upper()} | "
        f"Entry: ${price:.6f}"
    )
    
    if 'stop_loss' in kwargs:
        message += f" | SL: ${kwargs['stop_loss']:.6f}"
    
    if 'take_profit_1' in kwargs:
        message += f" | TP1: ${kwargs['take_profit_1']:.6f}"
    
    if 'leverage' in kwargs:
        message += f" | Leverage: {kwargs['leverage']}x"
    
    trading_logger.info(message)

def log_trade_result(pair: str, side: str, entry_price: float, 
                    exit_price: float, pnl_percent: float, reason: str):
    """
    Log trade result with standardized format
    
    Args:
        pair: Trading pair
        side: Trade side
        entry_price: Entry price
        exit_price: Exit price
        pnl_percent: Profit/loss percentage
        reason: Exit reason
    """
    
    trading_logger = logging.getLogger("TRADING")
    
    pnl_status = "PROFIT" if pnl_percent > 0 else "LOSS"
    
    message = (
        f"RESULT | {pnl_status} | {pair} | {side.upper()} | "
        f"Entry: ${entry_price:.6f} | Exit: ${exit_price:.6f} | "
        f"PnL: {pnl_percent:+.2f}% | Reason: {reason}"
    )
    
    trading_logger.info(message)

def log_performance_metric(metric_name: str, value: float, **kwargs):
    """
    Log performance metric
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        **kwargs: Additional context
    """
    
    perf_logger = logging.getLogger("PERFORMANCE")
    
    message = f"{metric_name}: {value:.4f}"
    
    if kwargs:
        context = " | ".join([f"{k}: {v}" for k, v in kwargs.items()])
        message += f" | {context}"
    
    perf_logger.info(message)

# Set up root logger configuration
def configure_root_logger():
    """Configure root logger to prevent unwanted output"""
    
    # Suppress noisy third-party loggers
    logging.getLogger('ccxt').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    
    # Configure requests logger
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    # Configure asyncio logger
    logging.getLogger('asyncio').setLevel(logging.WARNING)

# Initialize root logger configuration
configure_root_logger()
