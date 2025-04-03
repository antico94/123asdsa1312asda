import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from indicators.momentum import add_momentum_indicators
from indicators.trend import add_trend_indicators
from indicators.volatility import add_volatility_indicators, is_volatility_favorable
from indicators.wave_trend import add_wave_trend, is_wave_trend_signal
from indicators.patterns import detect_patterns
from core.price_levels import identify_support_resistance, identify_order_blocks
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
        self.price_levels = None
        self.bullish_blocks = []
        self.bearish_blocks = []
        self.last_update_time = None

    def _prepare_data(self, force_update=False):
        current_time = datetime.now()

        # Only update data every minute
        if not force_update and self.last_update_time and (current_time - self.last_update_time).seconds < 60:
            return self.cached_data

        # Get primary timeframe data
        primary_data = self.broker.get_historical_data(
            timeframe=self.config['PRIMARY_TIMEFRAME'],
            count=100
        )

        # Get confirmation timeframe data
        confirm_data = self.broker.get_historical_data(
            timeframe=self.config['CONFIRM_TIMEFRAME'],
            count=50
        )

        # Add indicators to primary timeframe data
        primary_data = add_volatility_indicators(primary_data, self.config)
        primary_data = add_trend_indicators(primary_data, self.config)
        primary_data = add_momentum_indicators(primary_data, self.config)
        primary_data = add_wave_trend(primary_data, self.config)

        # Add indicators to confirmation timeframe data
        confirm_data = add_trend_indicators(confirm_data, self.config)
        confirm_data = add_momentum_indicators(confirm_data, self.config)

        # Identify support and resistance levels
        self.price_levels = identify_support_resistance(
            primary_data,
            lookback=self.config['SUPRES_LOOKBACK']
        )

        # Identify order blocks
        self.bullish_blocks, self.bearish_blocks = identify_order_blocks(
            primary_data,
            strength=self.config['ORDER_BLOCK_STRENGTH']
        )

        # Store prepared data
        self.cached_data = {
            'primary': primary_data,
            'confirm': confirm_data
        }

        self.last_update_time = current_time
        return self.cached_data

    def get_trend_direction(self, data):
        # Use multiple indicators to determine trend
        points = 0

        # Check ADX for strong trend
        if 'adx' in data.columns and not pd.isna(data['adx'].iloc[-1]):
            adx_threshold = self.config['ADX_THRESHOLD'] * self.config.get('ADX_THRESHOLD_FACTOR', 0.8)
            if data['adx'].iloc[-1] > adx_threshold:
                # Check DI lines for direction
                if data['plus_di'].iloc[-1] > data['minus_di'].iloc[-1] * self.config.get('DI_COMPARISON_FACTOR', 0.9):
                    points += 1
                elif data['minus_di'].iloc[-1] > data['plus_di'].iloc[-1] * self.config.get('DI_COMPARISON_FACTOR',
                                                                                            0.9):
                    points -= 1

        # Check Ichimoku
        if 'senkou_span_a' in data.columns and not pd.isna(data['senkou_span_a'].iloc[-1]):
            # Price above cloud
            if (data['close'].iloc[-1] > data['senkou_span_a'].iloc[-1] and
                    data['close'].iloc[-1] > data['senkou_span_b'].iloc[-1]):
                points += 1
            # Price below cloud
            elif (data['close'].iloc[-1] < data['senkou_span_a'].iloc[-1] and
                  data['close'].iloc[-1] < data['senkou_span_b'].iloc[-1]):
                points -= 1

        # Check Supertrend
        if 'supertrend_direction' in data.columns and not pd.isna(data['supertrend_direction'].iloc[-1]):
            if data['supertrend_direction'].iloc[-1] > 0:
                points += 1
            else:
                points -= 1

        # Check Heiken Ashi
        if 'ha_open' in data.columns and not pd.isna(data['ha_open'].iloc[-1]):
            # Count consecutive bullish/bearish candles
            ha_bullish_count = 0
            ha_bearish_count = 0

            for i in range(1, min(5, len(data))):
                idx = -i
                if data['ha_close'].iloc[idx] > data['ha_open'].iloc[idx]:
                    ha_bullish_count += 1
                elif data['ha_close'].iloc[idx] < data['ha_open'].iloc[idx]:
                    ha_bearish_count += 1

            if ha_bullish_count >= 3:
                points += 1
            if ha_bearish_count >= 3:
                points -= 1

        # Trend classification
        if points >= 2:
            return 'uptrend'
        elif points <= -2:
            return 'downtrend'
        else:
            return 'ranging'

    def analyze_market(self):
        # Get latest market data with indicators
        data = self._prepare_data()
        primary_data = data['primary']
        confirm_data = data['confirm']

        # Reset signal points
        self.signal_points = {'buy': 0, 'sell': 0}

        # Check if volatility is favorable for trading
        if not is_volatility_favorable(primary_data, self.config):
            return self.signal_points

        # Check if current session is suitable for trading
        if not should_trade_in_current_session(self.config):
            return self.signal_points

        # Get trend direction
        primary_trend = self.get_trend_direction(primary_data)
        confirm_trend = self.get_trend_direction(confirm_data)

        # Check trend alignment
        if primary_trend == 'uptrend' and confirm_trend != 'downtrend':
            self.signal_points['buy'] += 1
        elif primary_trend == 'downtrend' and confirm_trend != 'uptrend':
            self.signal_points['sell'] += 1

        # Check RSI
        if 'rsi' in primary_data.columns and not pd.isna(primary_data['rsi'].iloc[-1]):
            if primary_data['rsi'].iloc[-1] < self.config['RSI_OVERSOLD']:
                self.signal_points['buy'] += 1
            elif primary_data['rsi'].iloc[-1] > self.config['RSI_OVERBOUGHT']:
                self.signal_points['sell'] += 1

        # Check Stochastic
        if 'stoch_k' in primary_data.columns and not pd.isna(primary_data['stoch_k'].iloc[-1]):
            # Bullish crossover in oversold
            if (primary_data['stoch_k'].iloc[-1] > primary_data['stoch_d'].iloc[-1] and
                    primary_data['stoch_k'].iloc[-2] < primary_data['stoch_d'].iloc[-2] and
                    primary_data['stoch_k'].iloc[-1] < 30):
                self.signal_points['buy'] += 1

            # Bearish crossover in overbought
            if (primary_data['stoch_k'].iloc[-1] < primary_data['stoch_d'].iloc[-1] and
                    primary_data['stoch_k'].iloc[-2] > primary_data['stoch_d'].iloc[-2] and
                    primary_data['stoch_k'].iloc[-1] > 70):
                self.signal_points['sell'] += 1

        # Check MACD
        if 'macd_line' in primary_data.columns and not pd.isna(primary_data['macd_line'].iloc[-1]):
            # Bullish crossover
            if (primary_data['macd_line'].iloc[-1] > primary_data['signal_line'].iloc[-1] and
                    primary_data['macd_line'].iloc[-2] < primary_data['signal_line'].iloc[-2]):
                self.signal_points['buy'] += 1

            # Bearish crossover
            if (primary_data['macd_line'].iloc[-1] < primary_data['signal_line'].iloc[-1] and
                    primary_data['macd_line'].iloc[-2] > primary_data['signal_line'].iloc[-2]):
                self.signal_points['sell'] += 1

        # Check WaveTrend
        wt_signal = is_wave_trend_signal(primary_data, self.config)
        if wt_signal['signal'] == 'buy':
            self.signal_points['buy'] += wt_signal['strength']
        elif wt_signal['signal'] == 'sell':
            self.signal_points['sell'] += wt_signal['strength']

        # Check candlestick patterns
        patterns = detect_patterns(primary_data, lookback=10, price_levels=self.price_levels)

        if patterns.get('bullish_rejection', False) or patterns.get('hammer', False) or patterns.get('morning_star',
                                                                                                     False):
            self.signal_points['buy'] += 1

        if patterns.get('bearish_rejection', False) or patterns.get('shooting_star', False) or patterns.get(
                'evening_star', False):
            self.signal_points['sell'] += 1

        if patterns.get('breakout_up', False) and primary_trend != 'downtrend':
            self.signal_points['buy'] += 1

        if patterns.get('breakout_down', False) and primary_trend != 'uptrend':
            self.signal_points['sell'] += 1

        # Check Bollinger Bands
        if 'bb_upper' in primary_data.columns and not pd.isna(primary_data['bb_upper'].iloc[-1]):
            # Price near lower band in uptrend
            if (primary_data['close'].iloc[-1] < primary_data['bb_lower'].iloc[-1] * 1.01 and
                    primary_trend == 'uptrend'):
                self.signal_points['buy'] += 1

            # Price near upper band in downtrend
            if (primary_data['close'].iloc[-1] > primary_data['bb_upper'].iloc[-1] * 0.99 and
                    primary_trend == 'downtrend'):
                self.signal_points['sell'] += 1

        return self.signal_points

    def should_open_position(self):
        # Analyze market to get current signal
        signals = self.analyze_market()

        # Reduced threshold from 7 to 6
        required_points = self.config.get('REQUIRED_POINTS', 6)

        # Reduced from 3 to 2
        min_point_difference = self.config.get('POINT_DIFFERENCE', 2)

        # Buy signal
        if (signals['buy'] >= required_points and
                signals['buy'] - signals['sell'] >= min_point_difference):
            return 'buy'

        # Sell signal
        if (signals['sell'] >= required_points and
                signals['sell'] - signals['buy'] >= min_point_difference):
            return 'sell'

        return None

    def get_entry_confirmation(self, signal_type):
        # Get latest data
        data = self._prepare_data(force_update=True)
        primary_data = data['primary']

        # Lookback for entry confirmation
        lookback = self.config['ENTRY_LOOKBACK']

        # Indicators to confirm entry
        confirmations = 0

        # Check recent price action
        trend = self.get_trend_direction(primary_data)
        if (signal_type == 'buy' and trend == 'uptrend') or (signal_type == 'sell' and trend == 'downtrend'):
            confirmations += 1

        # Check order blocks
        price = self.broker.get_current_price()
        current_price = price['ask'] if signal_type == 'buy' else price['bid']

        if signal_type == 'buy':
            # Check if price is near a bullish order block
            for block in self.bullish_blocks[-5:]:  # Check most recent blocks
                if block['low'] <= current_price <= block['high'] * 1.005:
                    confirmations += 1
                    break
        else:
            # Check if price is near a bearish order block
            for block in self.bearish_blocks[-5:]:
                if block['low'] * 0.995 <= current_price <= block['high']:
                    confirmations += 1
                    break

        # Check support/resistance levels
        if len(self.price_levels) > 0:
            for _, level_price, level_type in self.price_levels.values:
                if signal_type == 'buy' and level_type == 'support':
                    if abs(current_price - level_price) / level_price < 0.001:
                        confirmations += 1
                        break
                elif signal_type == 'sell' and level_type == 'resistance':
                    if abs(current_price - level_price) / level_price < 0.001:
                        confirmations += 1
                        break

        # Check recent patterns
        patterns = detect_patterns(primary_data, lookback=lookback)

        if signal_type == 'buy' and (patterns.get('bullish', False) or patterns.get('hammer', False)):
            confirmations += 1

        if signal_type == 'sell' and (patterns.get('bearish', False) or patterns.get('shooting_star', False)):
            confirmations += 1

        # Reduced from 2 to 1
        min_confirmations = self.config.get('MIN_CONFIRMATIONS', 1)

        return confirmations >= min_confirmations