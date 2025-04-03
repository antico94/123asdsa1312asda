"""
Momentum Indicators Module
Contains momentum indicators like RSI, MACD, Stochastic, and CCI
"""
import numpy as np
import pandas as pd


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)

    Args:
        df: DataFrame containing 'close' column
        period: RSI period

    Returns:
        Series containing RSI values
    """
    # Calculate price changes
    delta = df['close'].diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26,
                   signal_period: int = 9) -> pd.DataFrame:
    """
    Calculate Moving Average Convergence Divergence (MACD)

    Args:
        df: DataFrame containing 'close' column
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period

    Returns:
        DataFrame with MACD line and signal line
    """
    # Calculate the fast and slow EMAs
    fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()

    # Calculate the MACD line
    macd_line = fast_ema - slow_ema

    # Calculate the signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Calculate the histogram
    histogram = macd_line - signal_line

    result_df = pd.DataFrame({
        'macd_line': macd_line,
        'signal_line': signal_line,
        'histogram': histogram
    }, index=df.index)

    return result_df


def calculate_stochastic(df: pd.DataFrame, k_period: int = 5, d_period: int = 3, slowing: int = 3) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator

    Args:
        df: DataFrame containing 'high', 'low', 'close' columns
        k_period: %K period
        d_period: %D period
        slowing: %K slowing period

    Returns:
        DataFrame with %K and %D lines
    """
    # Calculate %K
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()

    # Apply slowing period if specified
    if slowing > 1:
        close_mean = df['close'].rolling(window=slowing).mean()
    else:
        close_mean = df['close']

    k = 100 * ((close_mean - low_min) / (high_max - low_min))

    # Calculate %D as a simple moving average of %K
    d = k.rolling(window=d_period).mean()

    result_df = pd.DataFrame({
        'stoch_k': k,
        'stoch_d': d
    }, index=df.index)

    return result_df


def calculate_cci(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Commodity Channel Index (CCI)

    Args:
        df: DataFrame containing 'high', 'low', 'close' columns
        period: CCI period

    Returns:
        Series containing CCI values
    """
    # Calculate typical price
    tp = (df['high'] + df['low'] + df['close']) / 3

    # Calculate the 20-period SMA of the typical price
    tp_sma = tp.rolling(window=period).mean()

    # Calculate the mean deviation
    mean_dev = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))

    # Calculate CCI
    cci = (tp - tp_sma) / (0.015 * mean_dev)

    return cci


def add_momentum_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add momentum indicators to DataFrame

    Args:
        df: Price DataFrame with OHLC data
        config: Configuration dictionary

    Returns:
        DataFrame with added indicators
    """
    # Copy the DataFrame to avoid modifying the original
    result_df = df.copy()

    # Add RSI
    if config.get('USE_RSI', True):
        result_df['rsi'] = calculate_rsi(df, period=config.get('RSI_PERIOD', 14))

    # Add MACD
    if config.get('USE_MACD', True):
        macd_df = calculate_macd(
            df,
            fast_period=config.get('MACD_FAST', 12),
            slow_period=config.get('MACD_SLOW', 26),
            signal_period=config.get('MACD_SIGNAL', 9)
        )
        result_df = pd.concat([result_df, macd_df], axis=1)

    # Add Stochastic
    if config.get('USE_STOCHASTIC', True):
        stoch_df = calculate_stochastic(
            df,
            k_period=config.get('STOCH_K', 5),
            d_period=config.get('STOCH_D', 3),
            slowing=config.get('STOCH_SLOWING', 3)
        )
        result_df = pd.concat([result_df, stoch_df], axis=1)

    # Add CCI
    if config.get('USE_CCI', True):
        result_df['cci'] = calculate_cci(df, period=config.get('CCI_PERIOD', 14))

    return result_df