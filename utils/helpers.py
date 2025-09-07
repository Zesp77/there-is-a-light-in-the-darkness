"""
Helper Functions for Lighter Point Farming Bot

Collection of utility functions for common operations like formatting,
calculations, validations, and data processing.
"""

import re
import math
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta
from typing import Union, Optional, List, Dict, Any
import pandas as pd
import numpy as np

def format_currency(amount: float, currency: str = "USD", decimals: int = 2) -> str:
    """
    Format currency amount with proper symbols and decimals
    
    Args:
        amount: Amount to format
        currency: Currency code (USD, USDT, etc.)
        decimals: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    try:
        if currency.upper() in ['USD', 'USDT', 'USDC']:
            symbol = '$'
        elif currency.upper() == 'BTC':
            symbol = '₿'
        elif currency.upper() == 'ETH':
            symbol = 'Ξ'
        else:
            symbol = currency + ' '
        
        # Handle large numbers with K/M/B notation
        if abs(amount) >= 1_000_000_000:
            return f"{symbol}{amount/1_000_000_000:.1f}B"
        elif abs(amount) >= 1_000_000:
            return f"{symbol}{amount/1_000_000:.1f}M"
        elif abs(amount) >= 1_000:
            return f"{symbol}{amount/1_000:.1f}K"
        else:
            return f"{symbol}{amount:,.{decimals}f}"
    
    except Exception:
        return f"{amount:.{decimals}f}"

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change (positive or negative)
    """
    try:
        if old_value == 0:
            return 0.0 if new_value == 0 else float('inf')
        
        return ((new_value - old_value) / old_value) * 100
    
    except (TypeError, ZeroDivisionError):
        return 0.0

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers with fallback for zero division
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    
    except (TypeError, ValueError):
        return default

def round_to_precision(value: float, precision: int = 8) -> float:
    """
    Round value to specified decimal places using banker's rounding
    
    Args:
        value: Value to round
        precision: Number of decimal places
        
    Returns:
        Rounded value
    """
    try:
        if pd.isna(value) or not isinstance(value, (int, float)):
            return 0.0
        
        decimal_value = Decimal(str(value))
        rounded = decimal_value.quantize(
            Decimal('0.1') ** precision,
            rounding=ROUND_HALF_UP
        )
        
        return float(rounded)
    
    except Exception:
        return 0.0

def validate_pair_format(pair: str) -> bool:
    """
    Validate trading pair format
    
    Args:
        pair: Trading pair string (e.g., "BTC/USDT", "ENA/USDT:USDT")
        
    Returns:
        True if format is valid
    """
    try:
        # Pattern for spot pairs: BASE/QUOTE
        spot_pattern = r'^[A-Z0-9]{2,10}\/[A-Z0-9]{2,10}$'
        
        # Pattern for futures pairs: BASE/QUOTE:SETTLEMENT
        futures_pattern = r'^[A-Z0-9]{2,10}\/[A-Z0-9]{2,10}:[A-Z0-9]{2,10}$'
        
        return (re.match(spot_pattern, pair) is not None or 
                re.match(futures_pattern, pair) is not None)
    
    except Exception:
        return False

def parse_pair(pair: str) -> Dict[str, str]:
    """
    Parse trading pair into components
    
    Args:
        pair: Trading pair string
        
    Returns:
        Dictionary with base, quote, and settlement currencies
    """
    try:
        if ':' in pair:
            # Futures pair: BASE/QUOTE:SETTLEMENT
            base_quote, settlement = pair.split(':')
            base, quote = base_quote.split('/')
            return {
                'base': base,
                'quote': quote, 
                'settlement': settlement,
                'type': 'futures'
            }
        else:
            # Spot pair: BASE/QUOTE
            base, quote = pair.split('/')
            return {
                'base': base,
                'quote': quote,
                'settlement': None,
                'type': 'spot'
            }
    
    except Exception:
        return {
            'base': '',
            'quote': '',
            'settlement': None,
            'type': 'unknown'
        }

def calculate_position_size(account_balance: float, risk_percent: float, 
                          entry_price: float, stop_loss: float, 
                          leverage: float = 1.0) -> float:
    """
    Calculate position size based on risk management
    
    Args:
        account_balance: Available balance
        risk_percent: Risk percentage (e.g., 2.0 for 2%)
        entry_price: Entry price
        stop_loss: Stop loss price
        leverage: Leverage multiplier
        
    Returns:
        Position size in base currency
    """
    try:
        # Calculate risk amount
        risk_amount = account_balance * (risk_percent / 100)
        
        # Calculate price risk per unit
        price_risk = abs(entry_price - stop_loss) / entry_price
        
        if price_risk == 0:
            return 0.0
        
        # Calculate position size
        position_size = risk_amount / (price_risk * entry_price)
        
        # Apply leverage
        position_size *= leverage
        
        return round_to_precision(position_size)
    
    except (TypeError, ZeroDivisionError):
        return 0.0

def calculate_pnl(entry_price: float, exit_price: float, position_size: float, 
                 side: str, leverage: float = 1.0) -> Dict[str, float]:
    """
    Calculate profit and loss for a trade
    
    Args:
        entry_price: Entry price
        exit_price: Exit price
        position_size: Position size
        side: Trade side ('long' or 'short')
        leverage: Leverage multiplier
        
    Returns:
        Dictionary with PnL information
    """
    try:
        if side.lower() == 'long':
            price_change = exit_price - entry_price
        else:  # short
            price_change = entry_price - exit_price
        
        # Calculate PnL
        pnl_absolute = (price_change / entry_price) * position_size * entry_price * leverage
        pnl_percent = (price_change / entry_price) * 100 * leverage
        
        return {
            'pnl_absolute': round_to_precision(pnl_absolute, 2),
            'pnl_percent': round_to_precision(pnl_percent, 4),
            'price_change': round_to_precision(price_change, 8),
            'price_change_percent': round_to_precision((price_change / entry_price) * 100, 4)
        }
    
    except Exception:
        return {
            'pnl_absolute': 0.0,
            'pnl_percent': 0.0,
            'price_change': 0.0,
            'price_change_percent': 0.0
        }

def format_timeframe(minutes: int) -> str:
    """
    Convert minutes to human-readable timeframe
    
    Args:
        minutes: Number of minutes
        
    Returns:
        Formatted timeframe string
    """
    try:
        if minutes < 60:
            return f"{minutes}m"
        elif minutes < 1440:
            hours = minutes // 60
            return f"{hours}h"
        else:
            days = minutes // 1440
            return f"{days}d"
    
    except Exception:
        return "unknown"

def timeframe_to_minutes(timeframe: str) -> int:
    """
    Convert timeframe string to minutes
    
    Args:
        timeframe: Timeframe string (e.g., '5m', '1h', '1d')
        
    Returns:
        Number of minutes
    """
    try:
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440
        else:
            return 5  # Default to 5 minutes
    
    except Exception:
        return 5

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio for a series of returns
    
    Args:
        returns: List of returns (percentages)
        risk_free_rate: Risk-free rate (annual percentage)
        
    Returns:
        Sharpe ratio
    """
    try:
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    except Exception:
        return 0.0

def calculate_max_drawdown(portfolio_values: List[float]) -> float:
    """
    Calculate maximum drawdown from portfolio values
    
    Args:
        portfolio_values: List of portfolio values over time
        
    Returns:
        Maximum drawdown percentage
    """
    try:
        if not portfolio_values or len(portfolio_values) < 2:
            return 0.0
        
        values = np.array(portfolio_values)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(values)
        
        # Calculate drawdown
        drawdown = (values - running_max) / running_max
        
        # Return maximum drawdown as positive percentage
        return abs(np.min(drawdown)) * 100
    
    except Exception:
        return 0.0

def is_market_hours() -> bool:
    """
    Check if current time is during active market hours (crypto is 24/7)
    
    Returns:
        True if markets are active (always True for crypto)
    """
    # Crypto markets are always open
    return True

def get_market_session() -> str:
    """
    Get current market session based on UTC time
    
    Returns:
        Market session name
    """
    try:
        current_hour = datetime.utcnow().hour
        
        if 0 <= current_hour < 8:
            return "Asia"
        elif 8 <= current_hour < 16:
            return "Europe"
        else:
            return "US"
    
    except Exception:
        return "Unknown"

def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate bot configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    try:
        # Required fields
        required_fields = [
            'timeframe',
            'exchange',
            'exchange.pair_whitelist'
        ]
        
        for field in required_fields:
            if '.' in field:
                # Nested field
                keys = field.split('.')
                current = config
                for key in keys:
                    if key not in current:
                        errors.append(f"Missing required field: {field}")
                        break
                    current = current[key]
            else:
                if field not in config:
                    errors.append(f"Missing required field: {field}")
        
        # Validate timeframe
        if 'timeframe' in config:
            valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
            if config['timeframe'] not in valid_timeframes:
                errors.append(f"Invalid timeframe: {config['timeframe']}")
        
        # Validate pairs
        if 'exchange' in config and 'pair_whitelist' in config['exchange']:
            pairs = config['exchange']['pair_whitelist']
            if not isinstance(pairs, list) or len(pairs) == 0:
                errors.append("pair_whitelist must be a non-empty list")
            else:
                for pair in pairs:
                    if not validate_pair_format(pair):
                        errors.append(f"Invalid pair format: {pair}")
        
        return errors
    
    except Exception as e:
        return [f"Configuration validation error: {str(e)}"]

def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate string to maximum length with suffix
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    try:
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
    
    except Exception:
        return str(text)[:max_length] if text else ""

def retry_on_exception(func, max_retries: int = 3, delay: float = 1.0, 
                      exceptions: tuple = (Exception,)):
    """
    Retry function on specified exceptions
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Function result or raises last exception
    """
    import time
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                time.sleep(delay)
            else:
                raise last_exception
    
    return None

def chunks(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    try:
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    except Exception:
        return []
