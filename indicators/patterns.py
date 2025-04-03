import numpy as np
import pandas as pd

def detect_engulfing(df, row_idx=-1):
    if row_idx <= 0 or row_idx >= len(df):
        row_idx = -1

    current_open = df['open'].iloc[row_idx]
    current_close = df['close'].iloc[row_idx]
    prior_open = df['open'].iloc[row_idx - 1]
    prior_close = df['close'].iloc[row_idx - 1]

    bullish = (prior_close < prior_open and
               current_close > current_open and
               current_close > prior_open and
               current_open < prior_close)

    bearish = (prior_close > prior_open and
               current_close < current_open and
               current_close < prior_open and
               current_open > prior_close)

    return {'bullish': bullish, 'bearish': bearish}

def detect_hammer_shooting_star(df, row_idx=-1):
    if row_idx <= 0 or row_idx >= len(df):
        row_idx = -1

    high = df['high'].iloc[row_idx]
    low = df['low'].iloc[row_idx]
    open_price = df['open'].iloc[row_idx]
    close = df['close'].iloc[row_idx]
    prior_low = df['low'].iloc[row_idx - 1]
    prior_high = df['high'].iloc[row_idx - 1]

    body_size = abs(close - open_price)
    full_size = high - low

    if full_size < 0.0001 or body_size < 0.0001:
        return {'hammer': False, 'shooting_star': False}

    upper_shadow = high - max(open_price, close)
    lower_shadow = min(open_price, close) - low

    hammer = (
            low < prior_low and
            lower_shadow > body_size * 2 and
            upper_shadow < lower_shadow * 0.3 and
            lower_shadow > full_size * 0.6
    )

    shooting_star = (
            high > prior_high and
            upper_shadow > body_size * 2 and
            lower_shadow < upper_shadow * 0.3 and
            upper_shadow > full_size * 0.6
    )

    return {'hammer': hammer, 'shooting_star': shooting_star}

def detect_doji(df, row_idx=-1, doji_threshold=0.1):
    if row_idx <= 0 or row_idx >= len(df):
        row_idx = -1

    high = df['high'].iloc[row_idx]
    low = df['low'].iloc[row_idx]
    open_price = df['open'].iloc[row_idx]
    close = df['close'].iloc[row_idx]

    body_size = abs(close - open_price)
    full_size = high - low

    if full_size < 0.0001:
        return False

    return body_size / full_size <= doji_threshold

def detect_morning_evening_star(df, row_idx=-1):
    if row_idx <= 2 or row_idx >= len(df):
        row_idx = -1

    first_open = df['open'].iloc[row_idx - 2]
    first_close = df['close'].iloc[row_idx - 2]
    middle_open = df['open'].iloc[row_idx - 1]
    middle_close = df['close'].iloc[row_idx - 1]
    last_open = df['open'].iloc[row_idx]
    last_close = df['close'].iloc[row_idx]

    first_body = abs(first_close - first_open)
    middle_body = abs(middle_close - middle_open)
    last_body = abs(last_close - last_open)

    morning_star = (
            first_close < first_open and
            abs(middle_close - middle_open) < abs(first_close - first_open) * 0.3 and
            last_close > last_open and
            last_close > (first_open + first_close) / 2
    )

    evening_star = (
            first_close > first_open and
            abs(middle_close - middle_open) < abs(first_close - first_open) * 0.3 and
            last_close < last_open and
            last_close < (first_open + first_close) / 2
    )

    return {'morning_star': morning_star, 'evening_star': evening_star}

def detect_breakout(df, row_idx=-1, lookback=10):
    if row_idx <= lookback or row_idx >= len(df):
        row_idx = -1

    current_close = df['close'].iloc[row_idx]

    prior_range = df.iloc[row_idx - lookback:row_idx]
    prior_high = prior_range['high'].max()
    prior_low = prior_range['low'].min()

    breakout_up = current_close > prior_high
    breakout_down = current_close < prior_low

    return {'breakout_up': breakout_up, 'breakout_down': breakout_down}

def detect_rejection(df, row_idx=-1, price_levels=None):
    if row_idx < 0 or row_idx >= len(df):
        row_idx = -1

    if price_levels is None or len(price_levels) == 0:
        return {'bullish_rejection': False, 'bearish_rejection': False}

    high = df['high'].iloc[row_idx]
    low = df['low'].iloc[row_idx]
    open_price = df['open'].iloc[row_idx]
    close = df['close'].iloc[row_idx]
    
    # Identify if price rejected from support or resistance
    bullish_rejection = False
    bearish_rejection = False
    
    # Find the nearest support/resistance levels
    for level in price_levels:
        level_price = level[1]  # price is stored at index 1
        level_type = level[2]   # type is stored at index 2
        
        # Check for bullish rejection at support
        if level_type == 'support':
            if low <= level_price * 1.001 and close > open_price:
                bullish_rejection = True
                break
                
        # Check for bearish rejection at resistance
        if level_type == 'resistance':
            if high >= level_price * 0.999 and close < open_price:
                bearish_rejection = True
                break
    
    return {'bullish_rejection': bullish_rejection, 'bearish_rejection': bearish_rejection}

def detect_patterns(df, lookback=10, price_levels=None):
    patterns = {}
    
    # We'll check for patterns at the last row
    row_idx = -1
    
    # Basic candlestick patterns
    patterns.update(detect_engulfing(df, row_idx))
    
    hammer_ss = detect_hammer_shooting_star(df, row_idx)
    patterns.update({
        'hammer': hammer_ss['hammer'],
        'shooting_star': hammer_ss['shooting_star']
    })
    
    patterns['doji'] = detect_doji(df, row_idx)
    
    star_patterns = detect_morning_evening_star(df, row_idx)
    patterns.update(star_patterns)
    
    # Price action patterns
    breakout = detect_breakout(df, row_idx, lookback)
    patterns.update(breakout)
    
    if price_levels is not None:
        rejection = detect_rejection(df, row_idx, price_levels)
        patterns.update(rejection)
    
    return patterns