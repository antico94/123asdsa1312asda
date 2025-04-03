import logging
import pandas as pd
from datetime import datetime, timedelta
from core.risk import calculate_position_size, calculate_stop_loss, calculate_take_profit
from core.session import should_trade_in_current_session


class PositionManager:
    def __init__(self, broker, config):
        self.broker = broker
        self.config = config
        self.open_positions = {}
        self.trades_history = []
        self.trades_today = 0
        self.last_trading_day = None
        self.logger = logging.getLogger('position_manager')

    def update(self):
        current_date = datetime.now().date()

        # Reset daily counter if new day
        if self.last_trading_day != current_date:
            self.trades_today = 0
            self.last_trading_day = current_date

        # Get latest positions from broker
        positions = self.broker.get_positions(magic_number=self.config['MAGIC_NUMBER'])

        # Update our tracking
        self.open_positions = {p.ticket: p for p in positions}

        # Check if any positions need trailing stop updates
        self._update_trailing_stops()

    def _update_trailing_stops(self):
        if not self.config['USE_TRAILING_STOP']:
            return

        current_prices = self.broker.get_current_price()

        for ticket, position in self.open_positions.items():
            if position.type == 'buy':
                # For buy positions, check if price has moved in our favor
                if current_prices['bid'] - position.open_price >= self.config['TRAILING_START'] * self.broker.pip_value:
                    # Calculate new stop loss level
                    new_stop = current_prices['bid'] - (self.config['TRAILING_STEP'] * self.broker.pip_value)

                    # Only move stop loss if it's higher than current stop
                    if position.stop_loss is None or new_stop > position.stop_loss:
                        self.broker.modify_position(ticket, stop_loss=new_stop)

            elif position.type == 'sell':
                # For sell positions, check if price has moved in our favor
                if position.open_price - current_prices['ask'] >= self.config['TRAILING_START'] * self.broker.pip_value:
                    # Calculate new stop loss level
                    new_stop = current_prices['ask'] + (self.config['TRAILING_STEP'] * self.broker.pip_value)

                    # Only move stop loss if it's lower than current stop
                    if position.stop_loss is None or new_stop < position.stop_loss:
                        self.broker.modify_position(ticket, stop_loss=new_stop)

    def can_open_new_position(self):
        # Check if we're already at max positions
        if len(self.open_positions) >= self.config['MAX_OPEN_TRADES']:
            return False

        # Check if we're in an allowed trading session
        if not should_trade_in_current_session(self.config):
            return False

        # Check if we've reached max trades for today
        if self.trades_today >= 10:  # Arbitrary limit to prevent overtrading
            return False

        # Check current spread
        current_price = self.broker.get_current_price()
        spread = self.broker.get_spread()

        if spread > self.config['MAX_SPREAD']:
            return False

        return True

    def open_position(self, signal_type, entry_price=None, stop_loss=None, take_profit=None):
        if not self.can_open_new_position():
            return None

        account_info = self.broker.get_account_info()

        # Determine stop loss if not provided
        if stop_loss is None:
            atr_data = self.broker.get_historical_data(timeframe=self.config['PRIMARY_TIMEFRAME'], count=20)
            from indicators.volatility import calculate_atr
            atr = calculate_atr(atr_data, period=self.config['ATR_PERIOD']).iloc[-1]

            current_price = self.broker.get_current_price()
            if signal_type == 'buy':
                entry = entry_price if entry_price is not None else current_price['ask']
                stop_loss = calculate_stop_loss(entry, 'buy', atr, 2.0)
            else:
                entry = entry_price if entry_price is not None else current_price['bid']
                stop_loss = calculate_stop_loss(entry, 'sell', atr, 2.0)

        # Calculate position size
        if signal_type == 'buy':
            entry = entry_price if entry_price is not None else self.broker.get_current_price()['ask']
            stop_pips = (entry - stop_loss) / self.broker.pip_value
        else:
            entry = entry_price if entry_price is not None else self.broker.get_current_price()['bid']
            stop_pips = (stop_loss - entry) / self.broker.pip_value

        volume = calculate_position_size(
            account_info,
            self.config['RISK_PERCENT'],
            stop_pips,
            self.broker.pip_value
        )

        # Determine take profit if not provided
        if take_profit is None:
            take_profit = calculate_take_profit(
                entry,
                stop_loss,
                self.config['RISK_REWARD'],
                signal_type
            )

        # Open the position with broker
        result = self.broker.open_position(
            order_type=signal_type,
            volume=volume,
            price=entry_price,  # None for market order
            stop_loss=stop_loss,
            take_profit=take_profit,
            magic_number=self.config['MAGIC_NUMBER'],
            comment=f"USDJPY_Bot_{signal_type}"
        )

        if result.success:
            self.trades_today += 1
            self.logger.info(
                f"Opened {signal_type} position at {entry}, SL: {stop_loss}, TP: {take_profit}, Vol: {volume}")
            return result.ticket
        else:
            self.logger.warning(f"Failed to open position: {result.message}")
            return None

    def close_position(self, ticket, reason="Manual close"):
        if ticket not in self.open_positions:
            return False

        result = self.broker.close_position(ticket)

        if result.success:
            position = self.open_positions.pop(ticket, None)

            # Record trade in history
            if position:
                trade_record = {
                    'ticket': ticket,
                    'type': position.type,
                    'open_time': position.open_time,
                    'close_time': datetime.now(),
                    'open_price': position.open_price,
                    'close_price': self.broker.get_current_price()['bid'] if position.type == 'buy' else
                    self.broker.get_current_price()['ask'],
                    'profit': position.profit_cash,
                    'volume': position.volume,
                    'reason': reason
                }

                self.trades_history.append(trade_record)
                self.logger.info(f"Closed position {ticket}, profit: {position.profit_cash:.2f}, reason: {reason}")

            return True
        else:
            self.logger.warning(f"Failed to close position {ticket}: {result.message}")
            return False

    def close_all_positions(self, reason="Close all"):
        tickets = list(self.open_positions.keys())
        success = True

        for ticket in tickets:
            if not self.close_position(ticket, reason):
                success = False

        return success

    def get_position_summary(self):
        total_profit = sum(p.profit_cash for p in self.open_positions.values())
        buy_count = sum(1 for p in self.open_positions.values() if p.type == 'buy')
        sell_count = sum(1 for p in self.open_positions.values() if p.type == 'sell')

        return {
            'total_positions': len(self.open_positions),
            'buy_positions': buy_count,
            'sell_positions': sell_count,
            'total_profit': total_profit,
            'trades_today': self.trades_today
        }