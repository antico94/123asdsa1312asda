"""
Volatility Indicators Module
Contains volatility-based indicators like ATR and Bollinger Bands
"""
import numpy as np
import pandas as pd


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)

    Args:
        df: DataFrame containing 'high', 'low', 'close' columns
        period: ATR period

    Returns:
        Series containing ATR values
    """
    # Calculate True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())

    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

    # Calculate ATR as moving average of TR
    atr = tr.rolling(window=period).mean()

    return atr


def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, deviation: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands

    Args:
        df: DataFrame containing 'close' column
        period: Bollinger Bands period
        deviation: Number of standard deviations for bands

    Returns:
        DataFrame with columns: 'bb_upper', 'bb_middle', 'bb_lower'
    """
    # Calculate middle band (SMA)
    middle_band = df['close'].rolling(window=period).mean()

    # Calculate standard deviation
    std_dev = df['close'].rolling(window=period).std()

    # Calculate upper and lower bands
    upper_band = middle_band + (std_dev * deviation)
    lower_band = middle_band - (std_dev * deviation)

    # Create result DataFrame
    bb_df = pd.DataFrame({
        'bb_upper': upper_band,
        'bb_middle': middle_band,
        'bb_lower': lower_band
    }, index=df.index)

    return bb_df


def is_volatility_favorable(df: pd.DataFrame, config: dict) -> bool:
    """
    Check if volatility conditions are favorable for trading

    Args:
        df: DataFrame with price data and indicators
        config: Configuration parameters

    Returns:
        True if volatility is favorable, False otherwise
    """
    if not config.get('USE_VOLATILITY_FILTER', True):
        return True

    # Check if ATR is above minimum threshold - reduced by 30%
    min_atr = config.get('MINIMUM_ATR_VALUE', 0.0005) * config.get('MINIMUM_ATR_MULTIPLIER', 0.7)
    if df['atr'].iloc[-1] < min_atr:
        return False

    # Check for excessive volatility (which could also be dangerous)
    # Increased from 3.0 to 3.5
    spike_threshold = config.get('ATR_SPIKE_THRESHOLD', 3.5)
    if df['atr'].iloc[-1] > df['atr'].iloc[-10] * spike_threshold:
        return False

    # Minimum required price movement check - reduced by 40%
    movement_factor = config.get('PRICE_MOVEMENT_FACTOR', 0.3)
    sufficient_movement = abs(df['close'].iloc[-1] - df['close'].iloc[-5]) > df['atr'].iloc[-1] * movement_factor

    return sufficient_movement


def add_volatility_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add volatility indicators to DataFrame

    Args:
        df: Price DataFrame with OHLC data
        config: Configuration dictionary

    Returns:
        DataFrame with added indicators
    """
    # Copy the DataFrame to avoid modifying the original
    result_df = df.copy()

    # Add ATR
    if config.get('USE_ATR', True):
        result_df['atr'] = calculate_atr(df, period=config.get('ATR_PERIOD', 14))

    # Add Bollinger Bands
    if config.get('USE_BOLLINGER_BANDS', True):
        bb_df = calculate_bollinger_bands(
            df,
            period=config.get('BB_PERIOD', 20),
            deviation=config.get('BB_DEVIATION', 2.0)
        )
        result_df = pd.concat([result_df, bb_df], axis=1)

    return result_df