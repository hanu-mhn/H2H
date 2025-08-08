"""
Helper utilities for the algo trading system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import asyncio
import aiohttp
from functools import wraps
import time


def retry_async(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Async retry decorator with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for exponential backoff
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    logging.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        return wrapper
    return decorator


def rate_limit(calls_per_second: float = 1.0):
    """
    Rate limiting decorator for API calls.
    
    Args:
        calls_per_second: Maximum calls allowed per second
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                await asyncio.sleep(left_to_wait)
            
            last_called[0] = time.time()
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that a DataFrame has required columns and data.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        True if valid, False otherwise
    """
    if df is None or df.empty:
        return False
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logging.error(f"Missing columns: {missing_columns}")
        return False
    
    # Check for sufficient data
    if len(df) < 50:  # Minimum data points for analysis
        logging.warning(f"Insufficient data: {len(df)} rows")
        return False
    
    return True


def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series
        method: 'simple' or 'log' returns
    
    Returns:
        Returns series
    """
    if method == 'simple':
        return prices.pct_change()
    elif method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError("Method must be 'simple' or 'log'")


def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling volatility.
    
    Args:
        returns: Returns series
        window: Rolling window size
    
    Returns:
        Volatility series
    """
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Risk-free rate (annualized)
    
    Returns:
        Sharpe ratio
    """
    excess_returns = returns.mean() * 252 - risk_free_rate  # Annualized
    return excess_returns / (returns.std() * np.sqrt(252))


def calculate_max_drawdown(prices: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown.
    
    Args:
        prices: Price series
    
    Returns:
        Tuple of (max_drawdown, start_date, end_date)
    """
    # Calculate running maximum
    running_max = prices.expanding().max()
    
    # Calculate drawdown
    drawdown = (prices - running_max) / running_max
    
    # Find maximum drawdown
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    
    # Find start of drawdown period
    start_idx = (running_max.loc[:max_dd_idx] == running_max.loc[max_dd_idx]).idxmax()
    
    return max_dd, start_idx, max_dd_idx


def normalize_symbol(symbol: str) -> str:
    """
    Normalize stock symbol for Yahoo Finance.
    
    Args:
        symbol: Raw stock symbol
    
    Returns:
        Normalized symbol
    """
    symbol = symbol.upper().strip()
    
    # Add .NS suffix for NSE stocks if not present
    if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
        symbol += '.NS'
    
    return symbol


def get_trading_sessions(start_date: str, end_date: str) -> List[datetime]:
    """
    Get list of trading sessions between dates (excluding weekends).
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        List of trading session dates
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    sessions = []
    current = start
    
    while current <= end:
        # Exclude weekends (Saturday=5, Sunday=6)
        if current.weekday() < 5:
            sessions.append(current)
        current += timedelta(days=1)
    
    return sessions


def format_currency(amount: float, currency: str = '₹') -> str:
    """
    Format amount as currency string.
    
    Args:
        amount: Amount to format
        currency: Currency symbol
    
    Returns:
        Formatted currency string
    """
    if abs(amount) >= 10000000:  # 1 crore
        return f"{currency}{amount/10000000:.2f}Cr"
    elif abs(amount) >= 100000:  # 1 lakh
        return f"{currency}{amount/100000:.2f}L"
    else:
        return f"{currency}{amount:.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage string.
    
    Args:
        value: Value to format (0.05 = 5%)
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def is_market_open() -> bool:
    """
    Check if Indian stock market is currently open.
    
    Returns:
        True if market is open, False otherwise
    """
    now = datetime.now()
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    
    # Market hours: 9:15 AM to 3:30 PM IST
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_open <= now <= market_close


def chunks(lst: List[Any], n: int) -> List[List[Any]]:
    """
    Yield successive n-sized chunks from list.
    
    Args:
        lst: List to chunk
        n: Chunk size
    
    Yields:
        Chunks of the list
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division by zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
    
    Returns:
        Result of division or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


async def fetch_with_session(session: aiohttp.ClientSession, url: str, **kwargs) -> Dict[str, Any]:
    """
    Fetch data from URL using aiohttp session.
    
    Args:
        session: aiohttp session
        url: URL to fetch
        **kwargs: Additional arguments for the request
    
    Returns:
        Response data
    """
    async with session.get(url, **kwargs) as response:
        response.raise_for_status()
        return await response.json()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean financial data DataFrame.
    
    Args:
        df: Raw data DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    # Remove duplicate indices
    df = df[~df.index.duplicated(keep='first')]
    
    # Forward fill missing values (common in financial data)
    df = df.fillna(method='ffill')
    
    # Remove any remaining NaN values
    df = df.dropna()
    
    # Ensure positive prices
    price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    for col in price_columns:
        if col in df.columns:
            df = df[df[col] > 0]
    
    return df
