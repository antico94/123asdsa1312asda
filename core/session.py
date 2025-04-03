from datetime import datetime, time
import pytz


def is_trading_session_active(current_time=None, optimize_for_london_tokyo=True, avoid_friday_ny=True):
    if current_time is None:
        current_time = datetime.now(pytz.UTC)

    day_of_week = current_time.weekday()  # 0 is Monday, 4 is Friday
    current_hour = current_time.hour

    # Weekend check
    if day_of_week >= 5:  # Saturday or Sunday
        return False

    # Avoid Friday New York afternoon if configured
    if avoid_friday_ny and day_of_week == 4 and current_hour >= 16:
        return False

    if optimize_for_london_tokyo:
        # London session: 8:00-16:00 UTC
        # Tokyo session: 0:00-9:00 UTC

        # London + Tokyo overlap: 8:00-9:00 UTC
        if 0 <= current_hour < 9 or 8 <= current_hour < 16:
            return True

        # Outside optimal sessions
        return False

    # Default trading hours (24/5 market)
    return True


def get_current_session(current_time=None):
    if current_time is None:
        current_time = datetime.now(pytz.UTC)

    current_hour = current_time.hour

    # Define session hours (UTC)
    sydney = (22, 7)  # 22:00 - 07:00
    tokyo = (0, 9)  # 00:00 - 09:00
    london = (8, 16)  # 08:00 - 16:00
    new_york = (13, 22)  # 13:00 - 22:00

    sessions = []

    # Check which sessions are active
    if sydney[0] <= current_hour or current_hour < sydney[1]:
        sessions.append('Sydney')

    if tokyo[0] <= current_hour < tokyo[1]:
        sessions.append('Tokyo')

    if london[0] <= current_hour < london[1]:
        sessions.append('London')

    if new_york[0] <= current_hour < new_york[1]:
        sessions.append('NewYork')

    if not sessions:
        return 'None'

    return '+'.join(sessions)


def is_high_volatility_session(current_time=None):
    if current_time is None:
        current_time = datetime.now(pytz.UTC)

    session = get_current_session(current_time)

    # London/New York overlap is typically highest volatility
    if 'London+NewYork' in session:
        return True

    # Individual sessions have different volatility profiles
    if 'London' in session:
        return True
    elif 'NewYork' in session:
        return True

    return False


def should_trade_in_current_session(config, current_time=None):
    if not config.get('USE_SESSION_FILTER', True):
        return True

    active = is_trading_session_active(
        current_time,
        optimize_for_london_tokyo=config.get('OPTIMIZE_FOR_LONDON_TOKYO', True),
        avoid_friday_ny=config.get('AVOID_FRIDAY_NY', True)
    )

    if not active:
        return False

    # If we're in a high volatility session, apply volatility multiplier
    session = get_current_session(current_time)

    if config.get('OPTIMIZE_FOR_LONDON_TOKYO', True) and ('London+Tokyo' in session):
        return True

    return active