"""
Candlestick Pattern Module
Implements recognition of key candlestick patterns
"""
import numpy as np
import pandas as pd


def detect_engulfing(df: pd.DataFrame, row_idx: int = -1) -> dict:
    """
    Detect bullish and bearish engulfing patterns

    Args:
        df: DataFrame with OHLC data
        row_idx: Index to check for pattern (default: last row)

    Returns:
        Dict with 'bullish' and 'bearish' keys and boolean values
    """
    if row_idx <= 0 or row_idx >= len(df):
        row_idx = -1  # Default to last row

    current_open = df['open'].iloc[row_idx]
    current_close = df['close'].iloc[row_idx]
    prior_open = df['open'].iloc[row_idx - 1]
    prior_close = df['close'].iloc[row_idx - 1]

    # Bullish engulfing
    bullish = (prior_close < prior_open and  # Prior candle is bearish
               current_close > current_open and  # Current candle is bullish
               current_close > prior_open and  # Close higher than prior open
               current_open < prior_close)  # Open lower than prior close

    # Bearish engulfing
    bearish = (prior_close > prior_open and  # Prior candle is bullish
               current_close < current_open and  # Current candle is bearish
               current_close < prior_open and  # Close lower than prior open
               current_open > prior_close)  # Open higher than prior close

    return {'bullish': bullish, 'bearish': bearish}


def detect_hammer_shooting_star(df: pd.DataFrame, row_idx: int = -1) -> dict:
    """
    Detect hammer (bullish) and shooting star (bearish) patterns

    Args:
        df: DataFrame with OHLC data
        row_idx: Index to check for pattern (default: last row)

    Returns:
        Dict with 'hammer' and 'shooting_star' keys and boolean values
    """
    if row_idx <= 0 or row_idx >= len(df):
        row_idx = -1  # Default to last row

    high = df['high'].iloc[row_idx]
    low = df['low'].iloc[row_idx]
    open_price = df['open'].iloc[row_idx]
    close = df['close'].iloc[row_idx]
    prior_low = df['low'].iloc[row_idx - 1]
    prior_high = df['high'].iloc[row_idx - 1]

    # Calculate candle parts
    body_size = abs(close - open_price)
    full_size = high - low

    # Skip if candle is too small
    if full_size < 0.0001 or body_size < 0.0001:
        return {'hammer': False, 'shooting_star': False}

    # Upper shadow
    upper_shadow = high - max(open_price, close)
    # Lower shadow
    lower_shadow = min(open_price, close) - low

    # Hammer criteria
    hammer = (
            low < prior_low and  # New low
            lower_shadow > body_size * 2 and  # Long lower shadow
            upper_shadow < lower_shadow * 0.3 and  # Short upper shadow
            lower_shadow > full_size * 0.6  # Lower shadow is significant part of candle
    )

    # Shooting star criteria
    shooting_star = (
            high > prior_high and  # New high
            upper_shadow > body_size * 2 and  # Long upper shadow
            lower_shadow < upper_shadow * 0.3 and  # Short lower shadow
            upper_shadow > full_size * 0.6  # Upper shadow is significant part of candle
    )

    return {'hammer': hammer, 'shooting_star': shooting_star}


def detect_doji(df: pd.DataFrame, row_idx: int = -1, doji_threshold: float = 0.1) -> bool:
    """
    Detect doji candlestick pattern

    Args:
        df: DataFrame with OHLC data
        row_idx: Index to check for pattern (default: last row)
        doji_threshold: Maximum body size relative to total size to qualify as doji

    Returns:
        True if pattern is detected, False otherwise
    """
    if row_idx <= 0 or row_idx >= len(df):
        row_idx = -1  # Default to last row

    high = df['high'].iloc[row_idx]
    low = df['low'].iloc[row_idx]
    open_price = df['open'].iloc[row_idx]
    close = df['close'].iloc[row_idx]

    # Calculate candle parts
    body_size = abs(close - open_price)
    full_size = high - low

    # Skip if candle is too small
    if full_size < 0.0001:
        return False

    # Doji has very small body compared to full size
    return body_size / full_size <= doji_threshold


def detect_morning_evening_star(df: pd.DataFrame, row_idx: int = -1) -> dict:
    """
    Detect morning star (bullish) and evening star (bearish) patterns

    Args:
        df: DataFrame with OHLC data
        row_idx: Index to check for pattern (default: last row)

    Returns:
        Dict with 'morning_star' and 'evening_star' keys and boolean values
    """
    if row_idx <= 2 or row_idx >= len(df):
        row_idx = -1  # Default to last row

    # Get candle data for the 3-candle pattern
    first_open = df['open'].iloc[row_idx - 2]
    first_close = df['close'].iloc[row_idx - 2]
    middle_open = df['open'].iloc[row_idx - 1]
    middle_close = df['close'].iloc[row_idx - 1]
    last_open = df['open'].iloc[row_idx]
    last_close = df['close'].iloc[row_idx]

    # Calculate body sizes
    first_body = abs(first_close - first_open)
    middle_body = abs(middle_close - middle_open)
    last_body = abs(last_close - last_open)

    # Morning star pattern
    morning_star = (
            first_close < first_open and  # First candle is bearish
            abs(middle_close - middle_open) < abs(first_close - first_open) * 0.3 and  # Middle candle is small
            last_close > last_open and  # Last candle is bullish
            last_close > (first_open + first_close) / 2  # Closed above midpoint of first candle
    )

    # Evening star pattern
    evening_star = (
            first_close > first_open and  # First candle is bullish
            abs(middle_close - middle_open) < abs(first_close - first_open) * 0.3 and  # Middle candle is small
            last_close < last_open and  # Last candle is bearish
            last_close < (first_open + first_close) / 2  # Closed below midpoint of first candle
    )

    return {'morning_star': morning_star, 'evening_star': evening_star}


def detect_breakout(df: pd.DataFrame, row_idx: int = -1, lookback: int = 10) -> dict:
    """
    Detect price breakouts from recent ranges

    Args:
        df: DataFrame with OHLC data
        row_idx: Index to check for breakout (default: last row)
        lookback: Number of bars to look back for range

    Returns:
        Dict with 'breakout_up' and 'breakout_down' keys and boolean values
    """
    if row_idx <= lookback or row_idx >= len(df):
        row_idx = -1  # Default to last row

    # Get current close
    current_close = df['close'].iloc[row_idx]

    # Calculate prior range
    prior_range = df.iloc[row_idx - lookback:row_idx]
    prior_high = prior_range['high'].max()
    prior_low = prior_range['low'].min()

    # Detect breakouts
    breakout_up = current_close > prior_high
    breakout_down = current_close < prior_low

    return {'breakout_up': breakout_up, 'breakout_down': breakout_down}


def detect_rejection(df: pd.DataFrame, row_idx: int = -1, price_levels: pd.DataFrame = None) -> dict:
    """
    Detect price rejection at support/resistance levels

    Args:
        df: DataFrame with OHLC data
        row_idx: Index to check for rejection (default: last row)
        price_levels: DataFrame with support and resistance levels

    Returns:
        Dict with 'bullish_rejection' and 'bearish_rejection' keys and boolean values
    """
    if row_idx < 0 or row_idx >= len(df):
        row_idx = -1  # Default to last row

    if price_levels is None or len(price_levels) == 0:
        return {'bullish_rejection': False, 'bearish_rejection': False}

    # Get candle data
    high = df['high'].iloc[row_idx]
    low = df['low'].iloc[row_idx]
    open_price = df