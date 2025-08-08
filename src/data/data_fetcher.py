"""
Data fetching module for Yahoo Finance API integration.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
import aiohttp
import logging
from concurrent.futures import ThreadPoolExecutor

from ..utils.config import Config
from ..utils.helpers import retry_async, rate_limit, normalize_symbol, clean_data
from ..utils.logger import setup_logger


class DataFetcher:
    """Yahoo Finance data fetcher with async support."""
    
    def __init__(self):
        """Initialize the data fetcher."""
        self.config = Config()
        self.logger = setup_logger(__name__)
        self.data_config = self.config.get_data_config()
        self.cache = {}
        self.cache_timeout = self.data_config.get('cache_duration', 3600)  # 1 hour
        
    @retry_async(max_retries=3)
    @rate_limit(calls_per_second=0.5)  # Respect Yahoo Finance rate limits
    async def fetch_stock_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch stock data for a single symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            DataFrame with stock data
        """
        symbol = normalize_symbol(symbol)
        
        # Check cache first
        cache_key = f"{symbol}_{period}_{interval}_{start_date}_{end_date}"
        if self._is_cache_valid(cache_key):
            self.logger.debug(f"Using cached data for {symbol}")
            return self.cache[cache_key]['data'].copy()
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                data = await loop.run_in_executor(
                    executor,
                    self._fetch_data_sync,
                    symbol, period, interval, start_date, end_date
                )
            
            if data.empty:
                raise ValueError(f"No data retrieved for {symbol}")
            
            # Clean and validate data
            data = clean_data(data)
            
            # Cache the data
            self.cache[cache_key] = {
                'data': data.copy(),
                'timestamp': datetime.now()
            }
            
            self.logger.info(f"Fetched {len(data)} data points for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def _fetch_data_sync(
        self,
        symbol: str,
        period: str,
        interval: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """Synchronous data fetching using yfinance."""
        ticker = yf.Ticker(symbol)
        
        if start_date and end_date:
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=True
            )
        else:
            data = ticker.history(
                period=period,
                interval=interval,
                auto_adjust=True,
                prepost=True
            )
        
        return data
    
    async def fetch_multiple_stocks(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        batch_size: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks concurrently.
        
        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
            start_date: Start date
            end_date: End date
            batch_size: Number of concurrent requests
        
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        # Process symbols in batches to avoid overwhelming the API
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            # Create tasks for current batch
            tasks = [
                self.fetch_stock_data(symbol, period, interval, start_date, end_date)
                for symbol in batch
            ]
            
            try:
                # Execute batch concurrently
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for symbol, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Failed to fetch {symbol}: {str(result)}")
                        continue
                    
                    if not result.empty:
                        results[symbol] = result
                    else:
                        self.logger.warning(f"Empty data for {symbol}")
                
                # Add delay between batches
                if i + batch_size < len(symbols):
                    await asyncio.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                continue
        
        self.logger.info(f"Successfully fetched data for {len(results)}/{len(symbols)} symbols")
        return results
    
    async def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get stock information and metadata.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with stock information
        """
        symbol = normalize_symbol(symbol)
        
        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                info = await loop.run_in_executor(
                    executor,
                    self._get_info_sync,
                    symbol
                )
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error fetching info for {symbol}: {str(e)}")
            return {}
    
    def _get_info_sync(self, symbol: str) -> Dict[str, Any]:
        """Synchronous stock info fetching."""
        ticker = yf.Ticker(symbol)
        return ticker.info
    
    async def get_realtime_price(self, symbol: str) -> Optional[float]:
        """
        Get real-time price for a stock.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Current price or None if unavailable
        """
        try:
            # Get 1-minute data for the last day
            data = await self.fetch_stock_data(
                symbol,
                period="1d",
                interval="1m"
            )
            
            if not data.empty:
                return float(data['Close'].iloc[-1])
            
        except Exception as e:
            self.logger.error(f"Error fetching real-time price for {symbol}: {str(e)}")
        
        return None
    
    def get_market_status(self) -> Dict[str, Any]:
        """
        Get current market status.
        
        Returns:
            Dictionary with market status information
        """
        now = datetime.now()
        
        # Check if it's a weekday
        is_weekday = now.weekday() < 5
        
        # Indian market hours: 9:15 AM to 3:30 PM IST
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        is_open = is_weekday and market_open <= now <= market_close
        
        # Calculate next market open/close
        if is_open:
            next_event = "close"
            next_time = market_close
        elif is_weekday and now < market_open:
            next_event = "open"
            next_time = market_open
        else:
            # Calculate next business day
            days_ahead = 1
            if now.weekday() == 4:  # Friday
                days_ahead = 3  # Skip to Monday
            elif now.weekday() == 5:  # Saturday
                days_ahead = 2  # Skip to Monday
            
            next_time = (now + timedelta(days=days_ahead)).replace(
                hour=9, minute=15, second=0, microsecond=0
            )
            next_event = "open"
        
        return {
            'is_open': is_open,
            'is_weekday': is_weekday,
            'current_time': now,
            'market_open': market_open,
            'market_close': market_close,
            'next_event': next_event,
            'next_time': next_time,
            'time_to_next': next_time - now
        }
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False
        
        cache_age = (datetime.now() - self.cache[cache_key]['timestamp']).total_seconds()
        return cache_age < self.cache_timeout
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        self.logger.info("Data cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.cache)
        total_size = sum(
            len(entry['data']) for entry in self.cache.values()
        )
        
        return {
            'total_entries': total_entries,
            'total_data_points': total_size,
            'cache_timeout': self.cache_timeout
        }
