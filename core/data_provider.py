"""
Enhanced Data Provider for Lighter Point Farming Bot

Integrates with Lighter.xyz API using the provided data retrieval guide.
Handles both live data fetching and backtesting data retrieval.
"""

import aiohttp
import asyncio
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from pathlib import Path
import time

from utils.logger import setup_logger

logger = setup_logger(__name__)

class LighterDataProvider:
    """Enhanced data provider for Lighter.xyz API integration"""
    
    def __init__(self, config: dict = None):
        self.base_url = "https://mainnet.zklighter.elliot.ai/api/v1"
        self.config = config or {}
        self.session = None
        self.market_cache = {}
        self.market_ids = {
            'ENA/USD:USD': 29,
            'LINK/USD:USD': 8,
            'AAVE/USD:USD': 27
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Cache settings
        self.cache_duration = config.get('data_sources', {}).get('cache_duration', 300)
        self.historical_data_cache = {}
        
        logger.info("🔌 Enhanced Lighter Data Provider initialized")
    
    async def initialize(self):
        """Initialize the data provider with HTTP session"""
        try:
            self.session = aiohttp.ClientSession()
            
            # Load market data
            await self._load_markets()
            
            logger.info("✅ Lighter Data Provider initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing data provider: {e}")
            raise
    
    async def _load_markets(self):
        """Load available markets from Lighter API"""
        try:
            url = f"{self.base_url}/orderBooks"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'order_books' in data:
                        for market in data['order_books']:
                            symbol = market.get('symbol', '')
                            market_id = market.get('market_id')
                            
                            if symbol and market_id:
                                # Convert to our pair format
                                if symbol in ['ENA', 'LINK', 'AAVE']:
                                    pair_format = f"{symbol}/USD:USD"
                                    self.market_ids[pair_format] = market_id
                                    
                        logger.info(f"📊 Loaded {len(self.market_ids)} markets")
                    else:
                        logger.warning("⚠️ No order_books found in API response")
                else:
                    logger.error(f"❌ Failed to load markets: HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ Error loading markets: {e}")
            # Use fallback market IDs
            logger.info("🔄 Using fallback market IDs")
    
    async def _rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_market_id(self, pair: str) -> Optional[int]:
        """Get market ID for trading pair"""
        return self.market_ids.get(pair)
    
    async def get_ohlcv_data(self, pair: str, timeframe: str = '5m', limit: int = 200) -> Optional[pd.DataFrame]:
        """Get recent OHLCV data for live trading"""
        try:
            market_id = self._get_market_id(pair)
            if not market_id:
                logger.error(f"❌ Market ID not found for pair: {pair}")
                return None
            
            # Calculate time range for recent data
            now = datetime.now()
            timeframe_minutes = self._timeframe_to_minutes(timeframe)
            start_time = now - timedelta(minutes=timeframe_minutes * limit)
            
            # Convert to timestamps
            start_ts = int(start_time.timestamp())
            end_ts = int(now.timestamp())
            
            await self._rate_limit()
            
            url = f"{self.base_url}/candlesticks"
            params = {
                'market_id': market_id,
                'resolution': timeframe,
                'start_timestamp': start_ts,
                'end_timestamp': end_ts,
                'count_back': min(limit, 5000)  # API limit
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'candlesticks' in data and data['candlesticks']:
                        df = self._process_candlestick_data(data['candlesticks'])
                        logger.debug(f"📈 Retrieved {len(df)} candles for {pair}")
                        return df
                    else:
                        logger.warning(f"⚠️ No candlestick data for {pair}")
                        return None
                else:
                    error_text = await response.text()
                    logger.error(f"❌ API error for {pair}: HTTP {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Error fetching OHLCV data for {pair}: {e}")
            return None
    
    async def get_historical_data_for_backtest(self, pair: str, timeframe: str, 
                                             start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get historical data for backtesting"""
        try:
            cache_key = f"{pair}_{timeframe}_{start_date}_{end_date}"
            
            # Check cache first
            if cache_key in self.historical_data_cache:
                cache_time, data = self.historical_data_cache[cache_key]
                if time.time() - cache_time < self.cache_duration:
                    logger.debug(f"📦 Using cached data for {pair}")
                    return data
            
            market_id = self._get_market_id(pair)
            if not market_id:
                logger.error(f"❌ Market ID not found for pair: {pair}")
                return None
            
            # Convert dates to timestamps
            start_ts = int(pd.to_datetime(start_date).timestamp())
            end_ts = int(pd.to_datetime(end_date).timestamp())
            
            # Calculate number of candles
            timeframe_seconds = self._timeframe_to_seconds(timeframe)
            count_back = min(int((end_ts - start_ts) / timeframe_seconds), 5000)
            
            await self._rate_limit()
            
            url = f"{self.base_url}/candlesticks"
            params = {
                'market_id': market_id,
                'resolution': timeframe,
                'start_timestamp': start_ts,
                'end_timestamp': end_ts,
                'count_back': count_back
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'candlesticks' in data and data['candlesticks']:
                        df = self._process_candlestick_data(data['candlesticks'])
                        
                        # Cache the result
                        self.historical_data_cache[cache_key] = (time.time(), df)
                        
                        logger.info(f"📈 Retrieved {len(df)} historical candles for {pair}")
                        return df
                    else:
                        logger.warning(f"⚠️ No historical data for {pair}")
                        return None
                else:
                    error_text = await response.text()
                    logger.error(f"❌ API error for {pair}: HTTP {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Error fetching historical data for {pair}: {e}")
            return None
    
    def _process_candlestick_data(self, candlesticks: List[dict]) -> pd.DataFrame:
        """Process raw candlestick data into DataFrame"""
        try:
            df = pd.DataFrame(candlesticks)
            
            if df.empty:
                return pd.DataFrame()
            
            # Convert timestamp from milliseconds to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Rename columns for consistency
            column_mapping = {
                'timestamp': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume1': 'volume'  # Use quote volume (USD)
            }
            
            # Select and rename columns
            df = df.rename(columns=column_mapping)
            
            # Ensure we have required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = df[required_columns]
            
            # Convert price columns to float
            price_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in price_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Remove any rows with NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error processing candlestick data: {e}")
            return pd.DataFrame()
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
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
        """Convert timeframe string to seconds"""
        return self._timeframe_to_minutes(timeframe) * 60
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            logger.info("🔌 Data provider session closed")


class MarketDataProvider(LighterDataProvider):
    """Alias for backward compatibility"""
    pass


async def get_market_data_provider(config: dict) -> LighterDataProvider:
    """Factory function to create and initialize market data provider"""
    try:
        provider = LighterDataProvider(config)
        await provider.initialize()
        return provider
    except Exception as e:
        logger.error(f"❌ Error creating market data provider: {e}")
        raise
