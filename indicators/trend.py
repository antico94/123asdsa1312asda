"""
Trend Indicators Module
Contains trend-following indicators like DMI/ADX, Ichimoku, Supertrend, and Heiken Ashi
"""
import numpy as np
import pandas as pd


def calculate_dmi_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Directional Movement Index (DMI) and Average Directional Index (ADX)

    Args:
        df: DataFrame containing 'high', 'low', 'close' columns
        period: DMI period

    Returns:
        DataFrame with columns: 'plus_di', 'minus_di', 'adx'
    """
    # Calculate True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))

    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

    # Smoothed True Range
    atr = tr.rolling(period).mean()

    # Plus Directional Movement (+DM)
    plus_dm = df['high'] - df['high'].shift(1)
    minus_dm = df['low'].shift(1) - df['low']

    # Conditions for +DM and -DM
    plus_dm = np.where(
        (plus_dm > 0) & (plus_dm > minus_dm),
        plus_dm,
        0
    )

    minus_dm = np.where(
        (minus_dm > 0) & (minus_dm > plus_dm),
        minus_dm,
        0
    )

    # Convert to pandas Series
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    # Smooth the +DM and -DM
    plus_dm_smoothed = plus_dm.rolling(period).mean()
    minus_dm_smoothed = minus_dm.rolling(period).mean()

    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smoothed / atr)
    minus_di = 100 * (minus_dm_smoothed / atr)

    # Calculate the Directional Index (DX)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

    # Calculate the Average Directional Index (ADX)
    adx = dx.rolling(period).mean()

    result_df = pd.DataFrame({
        'plus_di': plus_di,
        'minus_di': minus_di,
        'adx': adx
    }, index=df.index)

    return result_df


def calculate_ichimoku(df: pd.DataFrame,
                       tenkan_period: int = 9,
                       kijun_period: int = 26,
                       senkou_period: int = 52) -> pd.DataFrame:
    """
    Calculate Ichimoku Cloud indicator

    Args:
        df: DataFrame containing 'high', 'low', 'close' columns
        tenkan_period: Tenkan-sen period (conversion line)
        kijun_period: Kijun-sen period (base line)
        senkou_period: Senkou Span B period (leading span B)

    Returns:
        DataFrame with Ichimoku components
    """
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past 9 periods
    tenkan_sen = (df['high'].rolling(window=tenkan_period).max() +
                 df['low'].rolling(window=tenkan_period).min()) / 2

    # Calculate Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past 26 periods
    kijun_sen = (df['high'].rolling(window=kijun_period).max() +
                df['low'].rolling(window=kijun_period).min()) / 2

    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 shifted forward 26 periods
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)

    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for the past 52 periods, shifted forward 26 periods
    senkou_span_b = ((df['high'].rolling(window=senkou_period).max() +
                     df['low'].rolling(window=senkou_period).min()) / 2).shift(kijun_period)

    # Calculate Chikou Span (Lagging Span): Current closing price shifted backwards 26 periods
    chikou_span = df['close'].shift(-kijun_period)

    result_df = pd.DataFrame({
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    }, index=df.index)

    return result_df


def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Calculate Supertrend indicator

    Args:
        df: DataFrame containing 'high', 'low', 'close' columns
        period: ATR period
        multiplier: ATR multiplier

    Returns:
        DataFrame with 'supertrend' and 'supertrend_direction' columns
    """
    # Calculate ATR
    atr = calculate_atr(df, period)

    # Calculate basic upper and lower bands
    hl2 = (df['high'] + df['low']) / 2

    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    # Initialize Supertrend columns
    supertrend = pd.Series(0.0, index=df.index)
    supertrend_direction = pd.Series(0, index=df.index)

    # Set initial values
    supertrend.iloc[0] = lower_band.iloc[0] if df['close'].iloc[0] <= upper_band.iloc[0] else upper_band.iloc[0]
    supertrend_direction.iloc[0] = -1 if df['close'].iloc[0] <= upper_band.iloc[0] else 1

    # Calculate Supertrend values
    for i in range(1, len(df)):
        # Calculate new supertrend based on previous direction
        if supertrend_direction.iloc[i-1] == 1:  # Previous trend was up
            if df['close'].iloc[i] <= upper_band.iloc[i]:
                # Trend changes to down
                supertrend.iloc[i] = upper_band.iloc[i]
                supertrend_direction.iloc[i] = -1
            else:
                # Trend remains up, adjust supertrend value
                supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
                supertrend_direction.iloc[i] = 1
        else:  # Previous trend was down
            if df['close'].iloc[i] >= lower_band.iloc[i]:
                # Trend changes to up
                supertrend.iloc[i] = lower_band.iloc[i]
                supertrend_direction.iloc[i] = 1
            else:
                # Trend remains down, adjust supertrend value
                supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])
                supertrend_direction.iloc[i] = -1

    result_df = pd.DataFrame({
        'supertrend': supertrend,
        'supertrend_direction': supertrend_direction
    }, index=df.index)

    return result_df


def calculate_heiken_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Heiken Ashi candles

    Args:
        df: DataFrame containing 'open', 'high', 'low', 'close' columns

    Returns:
        DataFrame with Heiken Ashi OHLC values
    """
    ha_df = pd.DataFrame(index=df.index)

    # Calculate Heiken Ashi values
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    # Calculate first bar values
    ha_df['ha_open'] = df['open'].iloc[0]

    # Calculate remaining open values
    for i in range(1, len(df)):
        ha_df['ha_open'].iloc[i] = (ha_df['ha_open'].iloc[i-1] + ha_df['ha_close'].iloc[i-1]) / 2

    # Calculate high and low
    ha_df['ha_high'] = pd.concat([df['high'], ha_df['ha_open'], ha_df['ha_close']], axis=1).max(axis=1)
    ha_df['ha_low'] = pd.concat([df['low'], ha_df['ha_open'], ha_df['ha_close']], axis=1).min(axis=1)

    return ha_df


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


def add_trend_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add trend indicators to DataFrame

    Args:
        df: Price DataFrame with OHLC data
        config: Configuration dictionary

    Returns:
        DataFrame with added indicators
    """
    # Copy the DataFrame to avoid modifying the original
    result_df = df.copy()

    # Add DMI/ADX
    if config.get('USE_DMI', True):
        dmi_adx_df = calculate_dmi_adx(df, period=config.get('ADX_PERIOD', 14))
        result_df = pd.concat([result_df, dmi_adx_df], axis=1)

    # Add Ichimoku Cloud
    if config.get('USE_ICHIMOKU', True):
        ichimoku_df = calculate_ichimoku(
            df,
            tenkan_period=config.get('TENKAN_PERIOD', 9),
            kijun_period=config.get('KIJUN_PERIOD', 26),
            senkou_period=config.get('SENKOU_PERIOD', 52)
        )
        result_df = pd.concat([result_df, ichimoku_df], axis=1)

    # Add Supertrend
    if config.get('USE_SUPERTREND', True):
        supertrend_df = calculate_supertrend(
            df,
            period=config.get('SUPERTREND_PERIOD', 10),
            multiplier=config.get('SUPERTREND_MULTIPLIER', 3.0)
        )
        result_df = pd.concat([result_df, supertrend_df], axis=1)

    # Add Heiken Ashi
    if config.get('USE_HEIKEN', True):
        ha_df = calculate_heiken_ashi(df)
        result_df = pd.concat([result_df, ha_df], axis=1)

    return result_df