"""
Market Data Provider for Lighter Point Farming Bot

Handles data fetching from multiple sources including Lighter API,
fallback exchanges, and cached historical data for backtesting.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import logging
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import ccxt.async_support as ccxt
from pathlib import Path
from utils.logger import setup_logger


logger = setup_logger(__name__)

class MarketDataProvider:
    """
    Comprehensive market data provider supporting multiple sources
    """

    def __init__(self, config: dict):
        """
        Initialize market data provider
        
        Args:
            config: Configuration dictionary containing API keys and settings
        """
        self.config = config
        self.lighter_api_base = "https://mainnet.zklighter.elliot.ai/api/v1"
        self.lighter_websocket_url = "wss://mainnet.zklighter.elliot.ai/ws"
        
        # HTTP session for API calls
        self.session = None
        self.websocket = None
        
        # Fallback exchanges
        self.fallback_exchanges = {}
        
        # Data cache
        self.data_cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Historical data storage
        self.data_dir = Path("data/historical")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🔗 Market Data Provider initialized")

    async def initialize(self):
        """Initialize connections and fallback exchanges"""
        try:
            # Initialize HTTP session
            self.session = aiohttp.ClientSession()
            
            # Test Lighter API connection
            await self._test_lighter_connection()
            
            # Initialize fallback exchanges
            await self._initialize_fallback_exchanges()
            
            logger.info("✅ Market data provider initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing market data provider: {e}")
            raise

    async def _test_lighter_connection(self):
        """Test connection to Lighter API"""
        try:
            # Test basic API endpoint - use a public endpoint that doesn't require auth
            # Based on the API research, try a simpler endpoint first
            url = f"{self.lighter_api_base}/info"
            headers = self._get_lighter_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    logger.info("✅ Lighter API connection successful")
                    return True
                elif response.status == 401:
                    logger.warning("⚠️ Lighter API authentication failed - check API keys")
                    return False
                elif response.status == 400:
                    logger.warning("⚠️ Lighter API returned 400 - trying without authentication")
                    # Try without auth headers
                    async with self.session.get(url) as simple_response:
                        if simple_response.status == 200:
                            logger.info("✅ Lighter API connection successful (no auth)")
                            return True
                else:
                    logger.warning(f"⚠️ Lighter API returned status {response.status}")
                    return False
                    
        except Exception as e:
            logger.warning(f"⚠️ Lighter API connection failed: {e}")
            return False

async def _get_lighter_ohlcv(self, pair: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Get OHLCV data from Lighter API"""
    try:
        # For now, skip Lighter API and fall back to other exchanges
        # The Lighter API might need specific authentication or different endpoints
        logger.debug(f"⚠️ Skipping Lighter API for {pair} - falling back to other exchanges")
        return None
        
    except Exception as e:
        logger.debug(f"⚠️ Lighter OHLCV fetch failed for {pair}: {e}")
        return None

async def close(self):
    """Close connections and cleanup"""
    try:
        if self.session:
            await self.session.close()
            logger.debug("🔌 HTTP session closed")
            
        for exchange in self.fallback_exchanges.values():
            await exchange.close()
            logger.debug("🔌 Exchange connections closed")
            
        logger.info("🔌 Market data provider connections closed")
        
    except Exception as e:
        logger.error(f"❌ Error closing market data provider: {e}")


    def _get_lighter_headers(self) -> dict:
        """Get headers for Lighter API requests"""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'LighterScalpBot/1.0'
        }
        
        # Add API key if available
        data_sources = self.config.get('data_sources', {})
        api_key = data_sources.get('lighter_api_key')
        if api_key and api_key != 'YOUR_LIGHTER_API_KEY_HERE':
            headers['Authorization'] = f'Bearer {api_key}'
            
        return headers

    async def _initialize_fallback_exchanges(self):
        """Initialize fallback exchanges for data redundancy"""
        try:
            # Get data sources config
            data_sources = self.config.get('data_sources', {})
            fallback_exchanges = data_sources.get('fallback_exchanges', {})
            
            # Initialize Binance for fallback data
            binance_config = fallback_exchanges.get('binance', {})
            if binance_config.get('enabled', False):
                api_key = binance_config.get('api_key', '')
                secret = binance_config.get('secret', '')
                
                if api_key and secret and api_key != 'YOUR_BINANCE_API_KEY_HERE':
                    self.fallback_exchanges['binance'] = ccxt.binance({
                        'apiKey': api_key,
                        'secret': secret,
                        'sandbox': binance_config.get('sandbox', False),
                        'enableRateLimit': True,
                    })
                    
                    # Test fallback connections
                    for name, exchange in self.fallback_exchanges.items():
                        try:
                            await exchange.load_markets()
                            logger.info(f"✅ {name} fallback exchange initialized")
                        except Exception as e:
                            logger.warning(f"⚠️ {name} fallback exchange failed: {e}")
                else:
                    logger.warning("⚠️ Binance API keys not configured, fallback disabled")
                    
        except Exception as e:
            logger.error(f"❌ Error initializing fallback exchanges: {e}")

    async def get_current_price(self, pair: str) -> Optional[float]:
        """
        Get current price for a trading pair
        
        Args:
            pair: Trading pair (e.g., "ENA/USDT:USDT")
            
        Returns:
            Current price or None if unavailable
        """
        try:
            # Try Lighter API first
            price = await self._get_lighter_price(pair)
            if price:
                return price
                
            # Fallback to other exchanges
            return await self._get_fallback_price(pair)
            
        except Exception as e:
            logger.error(f"❌ Error getting current price for {pair}: {e}")
            return None

    async def _get_lighter_price(self, pair: str) -> Optional[float]:
        """Get price from Lighter API"""
        try:
            # Convert pair format for Lighter
            lighter_pair = self._convert_pair_format(pair, 'lighter')
            url = f"{self.lighter_api_base}/trades"
            params = {'symbol': lighter_pair, 'limit': 1}
            headers = self._get_lighter_headers()
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        return float(data[0].get('price', 0))
            return None
            
        except Exception as e:
            logger.debug(f"⚠️ Lighter API price fetch failed for {pair}: {e}")
            return None

    async def _get_fallback_price(self, pair: str) -> Optional[float]:
        """Get price from fallback exchanges"""
        try:
            for exchange_name, exchange in self.fallback_exchanges.items():
                try:
                    # Convert pair format for the exchange
                    exchange_pair = self._convert_pair_format(pair, exchange_name)
                    ticker = await exchange.fetch_ticker(exchange_pair)
                    if ticker and ticker['last']:
                        logger.debug(f"📊 Using {exchange_name} price for {pair}")
                        return float(ticker['last'])
                except Exception as e:
                    logger.debug(f"⚠️ {exchange_name} price fetch failed for {pair}: {e}")
                    continue
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting fallback price for {pair}: {e}")
            return None

    async def get_ohlcv_data(self, pair: str, timeframe: str = '5m', 
                           limit: int = 500) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data for analysis
        
        Args:
            pair: Trading pair
            timeframe: Timeframe (5m, 1h, etc.)
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Check cache first
            cache_key = f"{pair}_{timeframe}_{limit}"
            if self._is_cache_valid(cache_key):
                return self.data_cache[cache_key]['data']
            
            # Try Lighter API first
            data = await self._get_lighter_ohlcv(pair, timeframe, limit)
            if data is not None and not data.empty:
                self._cache_data(cache_key, data)
                return data
            
            # Fallback to other exchanges
            data = await self._get_fallback_ohlcv(pair, timeframe, limit)
            if data is not None and not data.empty:
                self._cache_data(cache_key, data)
                return data
            
            logger.warning(f"⚠️ No OHLCV data available for {pair}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting OHLCV data for {pair}: {e}")
            return None

    async def _get_lighter_ohlcv(self, pair: str, timeframe: str, 
                               limit: int) -> Optional[pd.DataFrame]:
        """Get OHLCV data from Lighter API"""
        try:
            lighter_pair = self._convert_pair_format(pair, 'lighter')
            
            # Convert timeframe to Lighter format
            resolution = self._timeframe_to_resolution(timeframe)
            
            # Calculate time range
            end_time = int(time.time())
            start_time = end_time - (limit * self._timeframe_to_seconds(timeframe))
            
            url = f"{self.lighter_api_base}/candles"  # Assuming this endpoint exists
            params = {
                'symbol': lighter_pair,
                'resolution': resolution,
                'from': start_time,
                'to': end_time
            }
            headers = self._get_lighter_headers()
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._format_ohlcv_data(data)
            return None
            
        except Exception as e:
            logger.debug(f"⚠️ Lighter OHLCV fetch failed for {pair}: {e}")
            return None

    async def _get_fallback_ohlcv(self, pair: str, timeframe: str, 
                                limit: int) -> Optional[pd.DataFrame]:
        """Get OHLCV data from fallback exchanges"""
        try:
            for exchange_name, exchange in self.fallback_exchanges.items():
                try:
                    exchange_pair = self._convert_pair_format(pair, exchange_name)
                    ohlcv = await exchange.fetch_ohlcv(
                        exchange_pair, timeframe, limit=limit
                    )
                    
                    if ohlcv:
                        df = pd.DataFrame(ohlcv, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume'
                        ])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        logger.debug(f"📊 Using {exchange_name} OHLCV for {pair}")
                        return df
                        
                except Exception as e:
                    logger.debug(f"⚠️ {exchange_name} OHLCV fetch failed for {pair}: {e}")
                    continue
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting fallback OHLCV for {pair}: {e}")
            return None

    def _convert_pair_format(self, pair: str, exchange: str) -> str:
        """Convert pair format for different exchanges"""
        try:
            # Original format: "ENA/USDT:USDT"
            if exchange == 'lighter':
                # Lighter might use different format
                return pair.replace('/', '_').replace(':', '')
            elif exchange == 'binance':
                # Binance format: "ENAUSDT" for spot, "ENA/USDT" for futures
                if ':' in pair:
                    # Futures pair
                    base_quote = pair.split(':')[0]
                    return base_quote  # "ENA/USDT"
                else:
                    # Spot pair
                    return pair.replace('/', '')
            else:
                return pair
                
        except Exception as e:
            logger.error(f"❌ Error converting pair format: {e}")
            return pair

    def _timeframe_to_resolution(self, timeframe: str) -> int:
        """Convert timeframe to resolution in minutes"""
        timeframe_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        return timeframe_map.get(timeframe, 5)

    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """Convert timeframe to seconds"""
        return self._timeframe_to_resolution(timeframe) * 60

    def _format_ohlcv_data(self, data: dict) -> pd.DataFrame:
        """Format OHLCV data from API response"""
        try:
            # Assuming data structure, adjust based on actual API response
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(data.get('t', []), unit='s'),
                'open': data.get('o', []),
                'high': data.get('h', []),
                'low': data.get('l', []),
                'close': data.get('c', []),
                'volume': data.get('v', [])
            })
            return df
        except Exception as e:
            logger.error(f"❌ Error formatting OHLCV data: {e}")
            return pd.DataFrame()

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.data_cache:
            return False
        cache_time = self.data_cache[cache_key]['timestamp']
        return (time.time() - cache_time) < self.cache_duration

    def _cache_data(self, cache_key: str, data: pd.DataFrame):
        """Cache data with timestamp"""
        self.data_cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }

    async def get_historical_data_for_backtest(self, pair: str, timeframe: str,
                                             start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Get historical data for backtesting
        
        Args:
            pair: Trading pair
            timeframe: Data timeframe
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Historical OHLCV DataFrame
        """
        try:
            # Check if we have cached historical data
            filename = f"{pair.replace('/', '_')}_{timeframe}_{start_date}_{end_date}.csv"
            filepath = self.data_dir / filename
            
            if filepath.exists():
                logger.info(f"📂 Loading cached historical data: {filename}")
                return pd.read_csv(filepath, parse_dates=['timestamp'])
            
            # Fetch historical data
            logger.info(f"🔍 Fetching historical data for {pair} ({start_date} to {end_date})")
            start_ts = int(pd.to_datetime(start_date).timestamp())
            end_ts = int(pd.to_datetime(end_date).timestamp())
            
            # Calculate required number of candles
            timeframe_seconds = self._timeframe_to_seconds(timeframe)
            total_candles = (end_ts - start_ts) // timeframe_seconds
            
            # Fetch data in chunks if needed
            data_chunks = []
            chunk_size = 1000  # Most APIs limit to 1000 candles per request
            
            for i in range(0, int(total_candles), chunk_size):
                chunk_start = start_ts + (i * timeframe_seconds)
                chunk_end = min(chunk_start + (chunk_size * timeframe_seconds), end_ts)
                
                chunk_data = await self._fetch_historical_chunk(
                    pair, timeframe, chunk_start, chunk_end
                )
                
                if chunk_data is not None and not chunk_data.empty:
                    data_chunks.append(chunk_data)
                
                # Rate limiting
                await asyncio.sleep(0.1)
            
            if not data_chunks:
                logger.warning(f"⚠️ No historical data available for {pair}")
                return None
            
            # Combine chunks
            historical_data = pd.concat(data_chunks, ignore_index=True)
            historical_data = historical_data.drop_duplicates(subset=['timestamp'])
            historical_data = historical_data.sort_values('timestamp')
            
            # Cache the data
            historical_data.to_csv(filepath, index=False)
            logger.info(f"💾 Cached historical data: {filename}")
            
            return historical_data
            
        except Exception as e:
            logger.error(f"❌ Error getting historical data for {pair}: {e}")
            return None

    async def _fetch_historical_chunk(self, pair: str, timeframe: str,
                                    start_ts: int, end_ts: int) -> Optional[pd.DataFrame]:
        """Fetch a chunk of historical data"""
        try:
            # Try Lighter API first
            data = await self._get_lighter_historical_chunk(pair, timeframe, start_ts, end_ts)
            if data is not None and not data.empty:
                return data
            
            # Fallback to other exchanges
            return await self._get_fallback_historical_chunk(pair, timeframe, start_ts, end_ts)
            
        except Exception as e:
            logger.debug(f"⚠️ Error fetching historical chunk: {e}")
            return None

    async def _get_lighter_historical_chunk(self, pair: str, timeframe: str,
                                          start_ts: int, end_ts: int) -> Optional[pd.DataFrame]:
        """Get historical chunk from Lighter API"""
        try:
            lighter_pair = self._convert_pair_format(pair, 'lighter')
            resolution = self._timeframe_to_resolution(timeframe)
            
            url = f"{self.lighter_api_base}/candles"
            params = {
                'symbol': lighter_pair,
                'resolution': resolution,
                'from': start_ts,
                'to': end_ts
            }
            headers = self._get_lighter_headers()
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._format_ohlcv_data(data)
            return None
            
        except Exception as e:
            logger.debug(f"⚠️ Lighter historical chunk failed: {e}")
            return None

    async def _get_fallback_historical_chunk(self, pair: str, timeframe: str,
                                           start_ts: int, end_ts: int) -> Optional[pd.DataFrame]:
        """Get historical chunk from fallback exchanges"""
        try:
            for exchange_name, exchange in self.fallback_exchanges.items():
                try:
                    exchange_pair = self._convert_pair_format(pair, exchange_name)
                    
                    # Calculate limit based on time range
                    timeframe_seconds = self._timeframe_to_seconds(timeframe)
                    limit = min(1000, (end_ts - start_ts) // timeframe_seconds)
                    
                    ohlcv = await exchange.fetch_ohlcv(
                        exchange_pair, timeframe, since=start_ts * 1000, limit=int(limit)
                    )
                    
                    if ohlcv:
                        df = pd.DataFrame(ohlcv, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume'
                        ])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        
                        # Filter to exact time range
                        start_dt = pd.to_datetime(start_ts, unit='s')
                        end_dt = pd.to_datetime(end_ts, unit='s')
                        df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
                        
                        return df
                        
                except Exception as e:
                    logger.debug(f"⚠️ {exchange_name} historical chunk failed: {e}")
                    continue
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting fallback historical chunk: {e}")
            return None

    async def close(self):
        """Close connections and cleanup"""
        try:
            if self.session:
                await self.session.close()
            for exchange in self.fallback_exchanges.values():
                await exchange.close()
            logger.info("🔌 Market data provider connections closed")
        except Exception as e:
            logger.error(f"❌ Error closing market data provider: {e}")


# Initialize singleton instance
_market_data_provider = None

async def get_market_data_provider(config: dict) -> MarketDataProvider:
    """Get singleton market data provider instance"""
    global _market_data_provider
    if _market_data_provider is None:
        _market_data_provider = MarketDataProvider(config)
        await _market_data_provider.initialize()
    return _market_data_provider
