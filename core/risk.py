import pandas as pd
from datetime import datetime, timedelta


def calculate_position_size(account_info, risk_percent, stop_loss_pips, pip_value):
    balance = account_info['balance']
    risk_amount = balance * (risk_percent / 100)

    if stop_loss_pips <= 0:
        return 0

    # Calculate the lot size based on risk
    pip_risk = stop_loss_pips * pip_value
    lot_size = risk_amount / pip_risk

    # Round to 2 decimal places (standard for most brokers)
    lot_size = round(lot_size, 2)

    # Ensure minimum and maximum lot sizes
    lot_size = max(0.01, min(lot_size, 10.0))

    return lot_size


def adjust_risk(account_info, base_risk_percent, trades_today, max_trades_per_day=5):
    if trades_today == 0:
        return base_risk_percent

    # Reduce risk as number of trades increases
    risk_factor = 1.0 - (trades_today / max_trades_per_day * 0.5)
    adjusted_risk = base_risk_percent * max(0.5, risk_factor)

    return adjusted_risk


def check_drawdown_limits(account_info, trades_history, config):
    equity = account_info['equity']
    balance = account_info['balance']

    # Check current drawdown
    current_dd = (balance - equity) / balance * 100

    # Check if we're exceeding daily drawdown
    if current_dd > config['MAX_DAILY_DRAWDOWN']:
        return False, "Daily drawdown limit exceeded"

    # Get trades from the past week
    one_week_ago = datetime.now() - timedelta(days=7)
    weekly_trades = [t for t in trades_history if t['close_time'] > one_week_ago]

    # Calculate weekly drawdown
    if weekly_trades:
        weekly_profit = sum(t['profit'] for t in weekly_trades)
        weekly_dd = abs(min(0, weekly_profit)) / balance * 100

        if weekly_dd > config['MAX_WEEKLY_DRAWDOWN']:
            return False, "Weekly drawdown limit exceeded"

    return True, "Risk limits OK"


def should_increase_risk(account_info, trades_history, base_risk):
    # Count recent winning trades
    recent_trades = trades_history[-10:] if len(trades_history) >= 10 else trades_history
    win_count = sum(1 for t in recent_trades if t['profit'] > 0)

    # If win rate is good, we can increase risk slightly
    if len(recent_trades) >= 5 and win_count / len(recent_trades) >= 0.7:
        return min(base_risk * 1.2, 2.0)

    return base_risk


def calculate_stop_loss(entry_price, trade_type, atr_value, multiplier=2.0):
    if trade_type == 'buy':
        return entry_price - (atr_value * multiplier)
    else:  # sell
        return entry_price + (atr_value * multiplier)


def calculate_take_profit(entry_price, stop_loss, risk_reward_ratio, trade_type):
    if trade_type == 'buy':
        stop_distance = entry_price - stop_loss
        take_profit = entry_price + (stop_distance * risk_reward_ratio)
    else:  # sell
        stop_distance = stop_loss - entry_price
        take_profit = entry_price - (stop_distance * risk_reward_ratio)

    return take_profit