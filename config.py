"""
Configuration settings for the USDJPY Trading System
"""
import pandas as pd

# Time Frame Settings
TIMEFRAME_M15 = "15min"
TIMEFRAME_H1 = "1H"
PRIMARY_TIMEFRAME = TIMEFRAME_M15
CONFIRM_TIMEFRAME = TIMEFRAME_H1

# Trading Framework
USE_SESSION_FILTER = True
USE_VOLATILITY_FILTER = True
MAX_SPREAD = 12  # in points
MAX_OPEN_TRADES = 2
ENTRY_LOOKBACK = 5  # bars to confirm entry

# Symbols
SYMBOL = "USDJPY"
PIP_VALUE = 0.01

# Price Action Indicators
USE_FRACTAL = True
USE_WAVE_TREND = True
WAVE_TREND_CHANNEL1 = 9
WAVE_TREND_CHANNEL2 = 12
WAVE_TREND_OVERSOLD = -60
WAVE_TREND_OVERBOUGHT = 60

# Price Structure Indicators
USE_ORDER_BLOCKS = True
ORDER_BLOCK_STRENGTH = 3  # bars lookback
USE_SUPRES = True
SUPRES_LOOKBACK = 20  # period

# Volume and Volatility
USE_BOLLINGER_BANDS = True
BB_PERIOD = 20
BB_DEVIATION = 2.0
USE_ATR = True
ATR_PERIOD = 14
MINIMUM_ATR_VALUE = 0.0005

# Trend Indicators
USE_ICHIMOKU = True
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_PERIOD = 52
USE_HEIKEN = True
USE_SUPERTREND = True
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
USE_DMI = True
ADX_PERIOD = 14
ADX_THRESHOLD = 25

# Momentum Indicators
USE_RSI = True
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
USE_MACD = True
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Oscillators
USE_STOCHASTIC = True
STOCH_K = 5
STOCH_D = 3
STOCH_SLOWING = 3
USE_CCI = True
CCI_PERIOD = 14

# Trade Management
RISK_PERCENT = 1.0  # risk per trade (%)
STOP_LOSS = 40  # in pips
RISK_REWARD = 2.0
USE_TRAILING_STOP = True
TRAILING_START = 40  # pips before trailing
TRAILING_STEP = 15  # trailing step in pips
USE_HEDGING = True

# Candlestick Pattern Settings
USE_CANDLE_PATTERNS = True
PATTERN_CONFIRMATION = 2  # bars to confirm pattern

# Session Optimization
OPTIMIZE_FOR_LONDON_TOKYO = True
AVOID_FRIDAY_NY = True

# Risk Management
DYNAMIC_LOT_MULTIPLIER = 1.0
USE_RISK_ADJUSTMENT = True
MAX_RISK_PERCENT = 1.5
MIN_RISK_PERCENT = 0.5
USE_EQUITY_PROTECTION = True
MAX_DAILY_DRAWDOWN = 3.0  # %
MAX_WEEKLY_DRAWDOWN = 5.0  # %

# Trading system adjustments (from our recent improvements)
# Reduced thresholds to improve win rate
ADX_THRESHOLD_FACTOR = 0.8  # 20% reduction
DI_COMPARISON_FACTOR = 0.9  # 10% less strict
MINIMUM_ATR_MULTIPLIER = 0.7  # 30% reduction
ATR_SPIKE_THRESHOLD = 3.5  # increased from 3.0
PRICE_MOVEMENT_FACTOR = 0.3  # reduced by 40% from 0.5
REQUIRED_POINTS = 6  # reduced from 7
POINT_DIFFERENCE = 2  # reduced from 3
MIN_CONFIRMATIONS = 1  # reduced from 2

# Magic number for identifying EA orders
MAGIC_NUMBER = 439721