"""
WaveTrend Oscillator Module
Implements the WaveTrend oscillator indicator
"""
import numpy as np
import pandas as pd


def calculate_wave_trend(df: pd.DataFrame,
                         channel1_period: int = 9,
                         channel2_period: int = 12) -> pd.DataFrame:
    """
    Calculate WaveTrend oscillator

    Args:
        df: DataFrame containing 'high', 'low', 'close' columns
        channel1_period: First smoothing period
        channel2_period: Second smoothing period

    Returns:
        DataFrame with 'wave_trend' and 'wave_trend_signal' columns
    """
    # Calculate HLC3 (typical price)
    hlc3 = (df['high'] + df['low'] + df['close']) / 3

    # First smoothing with SMA
    esa = hlc3.rolling(window=channel1_period).mean()

    # Calculate absolute deviation
    d = (hlc3 - esa).abs()

    # Second smoothing
    d_smoothed = d.rolling(window=channel2_period).mean()

    # Handle potential division by zero
    d_smoothed = d_smoothed.replace(0, np.nan)

    # Calculate WaveTrend
    ci = (hlc3 - esa) / (0.015 * d_smoothed)

    # Replace NaN values
    ci = ci.fillna(0)

    # Calculate WaveTrend signal line
    wave_trend_signal = ci.rolling(window=4).mean()

    result_df = pd.DataFrame({
        'wave_trend': ci,
        'wave_trend_signal': wave_trend_signal
    }, index=df.index)

    return result_df


def add_wave_trend(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add WaveTrend oscillator to DataFrame

    Args:
        df: Price DataFrame with OHLC data
        config: Configuration dictionary

    Returns:
        DataFrame with added WaveTrend indicator
    """
    # Copy the DataFrame to avoid modifying the original
    result_df = df.copy()

    # Add WaveTrend oscillator
    if config.get('USE_WAVE_TREND', True):
        wt_df = calculate_wave_trend(
            df,
            channel1_period=config.get('WAVE_TREND_CHANNEL1', 9),
            channel2_period=config.get('WAVE_TREND_CHANNEL2', 12)
        )
        result_df = pd.concat([result_df, wt_df], axis=1)

    return result_df


def is_wave_trend_signal(df: pd.DataFrame, config: dict, index: int = -1) -> dict:
    """
    Check if there's a WaveTrend signal at the given index

    Args:
        df: DataFrame with WaveTrend indicator
        config: Configuration parameters
        index: Index to check (default: last row)

    Returns:
        Dictionary with signal information
    """
    if not config.get('USE_WAVE_TREND', True) or 'wave_trend' not in df.columns:
        return {'signal': None, 'strength': 0}

    # Get the current and previous values
    wt_current = df['wave_trend'].iloc[index]
    wt_signal_current = df['wave_trend_signal'].iloc[index]

    wt_prev = df['wave_trend'].iloc[index - 1]
    wt_signal_prev = df['wave_trend_signal'].iloc[index - 1]

    # Get overbought/oversold thresholds
    wt_overbought = config.get('WAVE_TREND_OVERBOUGHT', 60)
    wt_oversold = config.get('WAVE_TREND_OVERSOLD', -60)

    signal = None
    strength = 0

    # Check for buy signal (oversold + cross up)
    if wt_current < wt_oversold and wt_prev < wt_signal_prev and wt_current > wt_signal_current:
        signal = 'buy'
        strength = 2

    # Check for sell signal (overbought + cross down)
    elif wt_current > wt_overbought and wt_prev > wt_signal_prev and wt_current < wt_signal_current:
        signal = 'sell'
        strength = 2

    return {'signal': signal, 'strength': strength}