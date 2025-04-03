import numpy as np
import pandas as pd


def identify_support_resistance(df, lookback=20, price_threshold=0.0005):
    levels = []

    for i in range(lookback, len(df)):
        current_price = df['close'].iloc[i]

        if _is_support(df, i, price_threshold):
            levels.append((df.index[i], current_price, 'support'))
        elif _is_resistance(df, i, price_threshold):
            levels.append((df.index[i], current_price, 'resistance'))

    return pd.DataFrame(levels, columns=['timestamp', 'price', 'type'])


def _is_support(df, i, threshold):
    if i <= 2 or i + 2 >= len(df):
        return False

    curr = df['low'].iloc[i]
    prev1 = df['low'].iloc[i - 1]
    prev2 = df['low'].iloc[i - 2]
    next1 = df['low'].iloc[i + 1]
    next2 = df['low'].iloc[i + 2]

    return (prev1 > curr < next1) and (prev2 > curr < next2)


def _is_resistance(df, i, threshold):
    if i <= 2 or i + 2 >= len(df):
        return False

    curr = df['high'].iloc[i]
    prev1 = df['high'].iloc[i - 1]
    prev2 = df['high'].iloc[i - 2]
    next1 = df['high'].iloc[i + 1]
    next2 = df['high'].iloc[i + 2]

    return (prev1 < curr > next1) and (prev2 < curr > next2)


def identify_order_blocks(df, strength=3):
    bullish_blocks = []
    bearish_blocks = []

    for i in range(strength + 1, len(df) - 1):
        # Bullish order block
        if df['close'].iloc[i] > df['close'].iloc[i - 1] and _is_uptrend_start(df, i, strength):
            bullish_blocks.append({
                'start_idx': i - strength,
                'end_idx': i - 1,
                'high': df['high'].iloc[i - strength:i].max(),
                'low': df['low'].iloc[i - strength:i].min(),
                'strength': strength
            })

        # Bearish order block
        if df['close'].iloc[i] < df['close'].iloc[i - 1] and _is_downtrend_start(df, i, strength):
            bearish_blocks.append({
                'start_idx': i - strength,
                'end_idx': i - 1,
                'high': df['high'].iloc[i - strength:i].max(),
                'low': df['low'].iloc[i - strength:i].min(),
                'strength': strength
            })

    return bullish_blocks, bearish_blocks


def _is_uptrend_start(df, i, strength):
    if i <= strength:
        return False

    # Check if price has been going down before this point
    downtrend_before = all(df['close'].iloc[j - 1] > df['close'].iloc[j] for j in range(i - strength, i - 1))

    # Check if price is now moving up
    uptrend_after = df['close'].iloc[i] > df['close'].iloc[i - 1]

    return downtrend_before and uptrend_after


def _is_downtrend_start(df, i, strength):
    if i <= strength:
        return False

    # Check if price has been going up before this point
    uptrend_before = all(df['close'].iloc[j - 1] < df['close'].iloc[j] for j in range(i - strength, i - 1))

    # Check if price is now moving down
    downtrend_after = df['close'].iloc[i] < df['close'].iloc[i - 1]

    return uptrend_before and downtrend_after


def is_near_level(price, levels, threshold=0.0005):
    for level in levels:
        level_price = level['price']
        diff = abs(price - level_price) / price
        if diff < threshold:
            return True, level
    return False, None