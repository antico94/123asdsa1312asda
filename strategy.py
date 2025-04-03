import pandas as pd
import numpy as np
import logging
from indicators.momentum import add_momentum_indicators
from indicators.trend import add_trend_indicators
from indicators.volatility import add_volatility_indicators, is_volatility_favorable
from indicators.wave_trend import add_wave_trend, is_wave_trend_signal
from indicators.patterns import detect_patterns
from core.price_levels import identify_support_resistance
from core.session import should_trade_in_current_session

class TradingStrategy:
    def __init__(self, broker, config):
        self.broker = broker
        self.config = config
        self.logger = logging.getLogger('trading_strategy')
        self.signal_points = {
            'buy': 0,
            'sell': 0
        }
        self.cached_data = None
        self.last_analyzed = None