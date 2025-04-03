import pandas as pd
import numpy as np
import logging
import time
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from broker_interface import BrokerInterface, Position, OrderResult
from strategy import TradingStrategy
from position_manager import PositionManager


class HistoricalDataProvider:
    def __init__(self, symbol, timeframes, start_date, end_date):
        self.symbol = symbol
        self.timeframes = timeframes
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        self.current_idx = {}
        self.logger = logging.getLogger('historical_data')

    def load_data(self, data_dir="historical_data"):
        for timeframe in self.timeframes:
            filename = f"{data_dir}/{self.symbol}_{timeframe}.csv"
            if os.path.exists(filename):
                df = pd.read_csv(filename, parse_dates=['datetime'], index_col='datetime')

                # Filter data by date range
                mask = (df.index >= self.start_date) & (df.index <= self.end_date)
                df = df.loc[mask]

                # Sort by datetime
                df = df.sort_index()

                self.data[timeframe] = df
                self.current_idx[timeframe] = 0
                self.logger.info(f"Loaded {len(df)} {timeframe} candles for {self.symbol}")
            else:
                self.logger.error(f"Historical data file not found: {filename}")
                raise FileNotFoundError(f"Data file {filename} not found")

    def generate_synthetic_data(self, years=10):
        for timeframe in self.timeframes:
            # Determine candle duration
            if timeframe.endswith('min'):
                minutes = int(timeframe[:-3])
                candle_duration = timedelta(minutes=minutes)
            elif timeframe.endswith('H'):
                hours = int(timeframe[:-1])
                candle_duration = timedelta(hours=hours)
            elif timeframe.endswith('D'):
                days = int(timeframe[:-1])
                candle_duration = timedelta(days=days)
            else:
                self.logger.error(f"Unsupported timeframe format: {timeframe}")
                continue

            # Generate dates
            current_date = self.start_date
            dates = []
            while current_date <= self.end_date:
                # Skip weekends for forex
                if current_date.weekday() < 5:  # Monday to Friday
                    dates.append(current_date)
                current_date += candle_duration

            # Generate synthetic price data
            np.random.seed(42)  # For reproducibility

            # Start with a price around 110 (typical for USDJPY)
            initial_price = 110.0

            # Use a random walk with drift
            daily_returns = np.random.normal(0.00005, 0.001, len(dates))  # slight upward drift

            # Calculate prices
            prices = initial_price * (1 + np.cumsum(daily_returns))

            # Generate OHLC data with some randomness
            data = []
            for i, date in enumerate(dates):
                close_price = prices[i]
                spread = 0.02  # typical for USDJPY

                # Add some randomness to open/high/low
                open_price = close_price * np.random.uniform(0.998, 1.002)
                high_price = max(open_price, close_price) * np.random.uniform(1.0005, 1.002)
                low_price = min(open_price, close_price) * np.random.uniform(0.998, 0.9995)

                # Add some volatility
                volume = np.random.exponential(100)

                data.append({
                    'datetime': date,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })

            # Create DataFrame
            df = pd.DataFrame(data)
            df.set_index('datetime', inplace=True)

            self.data[timeframe] = df
            self.current_idx[timeframe] = 0
            self.logger.info(f"Generated {len(df)} synthetic {timeframe} candles for {self.symbol}")

    def get_current_data(self, timeframe, lookback=100):
        if timeframe not in self.data:
            raise ValueError(f"No data available for timeframe {timeframe}")

        idx = self.current_idx[timeframe]
        if idx >= len(self.data[timeframe]):
            return None  # End of data

        start_idx = max(0, idx - lookback + 1)
        return self.data[timeframe].iloc[start_idx:idx + 1]

    def advance_time(self, steps=1):
        for timeframe in self.timeframes:
            self.current_idx[timeframe] = min(
                self.current_idx[timeframe] + steps,
                len(self.data[timeframe]) - 1
            )

    def get_current_time(self):
        # Use the primary timeframe for current time
        primary_tf = self.timeframes[0]
        idx = self.current_idx[primary_tf]
        if idx >= len(self.data[primary_tf]):
            return None
        return self.data[primary_tf].index[idx]

    def get_current_price(self):
        primary_tf = self.timeframes[0]
        idx = self.current_idx[primary_tf]
        if idx >= len(self.data[primary_tf]):
            return None

        close_price = self.data[primary_tf]['close'].iloc[idx]
        # Simulate a bid/ask spread
        bid = close_price - 0.01
        ask = close_price + 0.01

        return {"bid": bid, "ask": ask}


class BacktestBroker(BrokerInterface):
    def __init__(self, data_provider, initial_balance=10000.0, pip_value=0.01):
        self.data_provider = data_provider
        self.balance = initial_balance
        self.equity = initial_balance
        self.positions = {}
        self.next_ticket = 1000
        self.pip_value = pip_value
        self.symbol = data_provider.symbol
        self.trade_history = []

        self.logger = logging.getLogger('backtest_broker')

    def get_account_info(self):
        return {
            "balance": self.balance,
            "equity": self.equity,
            "margin_level": 100.0,
            "free_margin": self.balance * 0.9
        }

    def get_current_price(self, symbol=None):
        return self.data_provider.get_current_price()

    def get_historical_data(self, symbol=None, timeframe="15min", count=100):
        return self.data_provider.get_current_data(timeframe, count)

    def open_position(self, order_type, volume, price=None, stop_loss=None, take_profit=None, magic_number=None,
                      comment=""):
        current_price = self.get_current_price()
        if current_price is None:
            return OrderResult(success=False, message="No price data available")

        if price is None:
            # Market order
            price = current_price["ask"] if order_type == "buy" else current_price["bid"]

        ticket = self.next_ticket
        self.next_ticket += 1

        position = Position(
            ticket=ticket,
            symbol=self.symbol,
            type=order_type,
            volume=volume,
            open_price=price,
            open_time=self.data_provider.get_current_time(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            magic_number=magic_number,
            comment=comment
        )

        self.positions[ticket] = position
        self.logger.info(f"Opened {order_type.upper()} position: {ticket} at {price:.5f}, volume: {volume:.2f}")

        # Record trade in history
        trade_record = {
            'ticket': ticket,
            'type': order_type,
            'open_time': position.open_time,
            'open_price': price,
            'volume': volume,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        self.trade_history.append(trade_record)

        return OrderResult(success=True, ticket=ticket, message="Position opened successfully")

    def modify_position(self, ticket, stop_loss=None, take_profit=None):
        if ticket not in self.positions:
            return OrderResult(success=False, message=f"Position {ticket} not found")

        position = self.positions[ticket]

        if stop_loss is not None:
            position.stop_loss = stop_loss

        if take_profit is not None:
            position.take_profit = take_profit

        return OrderResult(success=True, message="Position modified successfully")

    def close_position(self, ticket, partial_volume=None):
        if ticket not in self.positions:
            return OrderResult(success=False, message=f"Position {ticket} not found")

        position = self.positions[ticket]
        current_price = self.get_current_price()

        if current_price is None:
            return OrderResult(success=False, message="No price data available")

        close_price = current_price["bid"] if position.type == "buy" else current_price["ask"]

        # Calculate profit/loss
        pip_diff = 0
        if position.type == "buy":
            pip_diff = (close_price - position.open_price) / self.pip_value
        else:
            pip_diff = (position.open_price - close_price) / self.pip_value

        profit = pip_diff * position.volume * 10  # Assume $10 per pip per standard lot

        # Update account balance
        self.balance += profit

        # Update trade history
        for trade in self.trade_history:
            if trade['ticket'] == ticket and 'close_price' not in trade:
                trade['close_time'] = self.data_provider.get_current_time()
                trade['close_price'] = close_price
                trade['profit'] = profit
                break

        # Remove position
        del self.positions[ticket]

        self.logger.info(f"Closed position {ticket} at {close_price:.5f}, profit: {profit:.2f}")

        return OrderResult(success=True, message=f"Position closed with profit: {profit:.2f}")

    def get_positions(self, magic_number=None):
        if magic_number is not None:
            return [p for p in self.positions.values() if p.magic_number == magic_number]
        return list(self.positions.values())

    def get_spread(self, symbol=None):
        current_price = self.get_current_price()
        if current_price is None:
            return 20  # Default spread
        return (current_price["ask"] - current_price["bid"]) / self.pip_value

    def update_positions(self):
        current_price = self.get_current_price()
        if current_price is None:
            return

        # Update profit calculations and check for SL/TP hits
        tickets_to_close = []

        for ticket, position in self.positions.items():
            # Calculate current profit
            if position.type == "buy":
                price_diff = current_price["bid"] - position.open_price
                position.profit_pips = price_diff / self.pip_value
            else:
                price_diff = position.open_price - current_price["ask"]
                position.profit_pips = price_diff / self.pip_value

            position.profit_cash = position.profit_pips * position.volume * 10

            # Check for stop loss hit
            if position.stop_loss is not None:
                if (position.type == "buy" and current_price["bid"] <= position.stop_loss) or \
                        (position.type == "sell" and current_price["ask"] >= position.stop_loss):
                    tickets_to_close.append((ticket, "Stop loss"))

            # Check for take profit hit
            if position.take_profit is not None:
                if (position.type == "buy" and current_price["bid"] >= position.take_profit) or \
                        (position.type == "sell" and current_price["ask"] <= position.take_profit):
                    tickets_to_close.append((ticket, "Take profit"))

        # Close positions that hit SL/TP
        for ticket, reason in tickets_to_close:
            self.close_position(ticket)
            self.logger.info(f"Position {ticket} closed: {reason}")

        # Update equity
        self.equity = self.balance + sum(p.profit_cash for p in self.positions.values())


class Backtester:
    def __init__(self, symbol, timeframes, start_date, end_date, config, initial_balance=10000.0):
        self.symbol = symbol
        self.timeframes = timeframes
        self.start_date = start_date
        self.end_date = end_date
        self.config = config
        self.initial_balance = initial_balance

        self.logger = logging.getLogger('backtester')

        # Initialize data provider
        self.data_provider = HistoricalDataProvider(symbol, timeframes, start_date, end_date)

        # Initialize broker
        self.broker = BacktestBroker(self.data_provider, initial_balance)

        # Initialize strategy and position manager
        self.strategy = TradingStrategy(self.broker, self.config)
        self.position_manager = PositionManager(self.broker, self.config)

        # For storing results
        self.equity_curve = []
        self.trade_results = []

    def load_data(self, data_dir="historical_data"):
        try:
            self.data_provider.load_data(data_dir)
            return True
        except FileNotFoundError:
            self.logger.warning("Historical data not found, generating synthetic data")
            self.data_provider.generate_synthetic_data(years=10)
            return True

    def run(self, progress_callback=None):
        self.logger.info(f"Starting backtest for {self.symbol} from {self.start_date} to {self.end_date}")
        self.logger.info(f"Initial balance: ${self.initial_balance:.2f}")

        total_steps = sum(len(self.data_provider.data[tf]) for tf in self.timeframes)
        steps_completed = 0

        # Track account changes
        self.equity_curve = []

        start_time = time.time()

        while True:
            # Get current time
            current_time = self.data_provider.get_current_time()
            if current_time is None:
                break  # End of data

            # Update positions
            self.broker.update_positions()
            self.position_manager.update()

            # Check for trading signals
            if self.position_manager.can_open_new_position():
                signal = self.strategy.should_open_position()

                if signal:
                    if self.strategy.get_entry_confirmation(signal):
                        self.logger.info(f"[{current_time}] Confirmed {signal} signal, opening position")
                        self.position_manager.open_position(signal)
                    else:
                        self.logger.debug(f"[{current_time}] Signal {signal} detected but confirmation failed")

            # Record account equity
            account_info = self.broker.get_account_info()
            self.equity_curve.append({
                'date': current_time,
                'equity': account_info['equity'],
                'balance': account_info['balance']
            })

            # Advance to next candle
            self.data_provider.advance_time()

            # Update progress
            steps_completed += 1
            if progress_callback and steps_completed % 100 == 0:
                progress_callback(steps_completed / total_steps)

        end_time = time.time()

        # Record final account state
        final_balance = self.broker.get_account_info()['balance']
        profit = final_balance - self.initial_balance
        profit_pct = (profit / self.initial_balance) * 100

        self.logger.info(f"Backtest completed in {end_time - start_time:.2f} seconds")
        self.logger.info(f"Final balance: ${final_balance:.2f}")
        self.logger.info(f"Total profit: ${profit:.2f} ({profit_pct:.2f}%)")

        # Analyze trades
        self.analyze_results()

        return {
            'initial_balance': self.initial_balance,
            'final_balance': final_balance,
            'profit': profit,
            'profit_pct': profit_pct,
            'equity_curve': pd.DataFrame(self.equity_curve),
            'trade_results': self.trade_results
        }

    def analyze_results(self):
        # Analyze completed trades
        trades = [t for t in self.broker.trade_history if 'close_price' in t]

        if not trades:
            self.logger.warning("No completed trades to analyze")
            return

        # Convert to DataFrame for easier analysis
        trades_df = pd.DataFrame(trades)
        trades_df['duration'] = trades_df['close_time'] - trades_df['open_time']

        # Calculate basic statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit'] > 0])
        losing_trades = len(trades_df[trades_df['profit'] <= 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        # Profit stats
        total_profit = trades_df['profit'].sum()
        avg_profit = trades_df[trades_df['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['profit'] <= 0]['profit'].mean() if losing_trades > 0 else 0

        # Risk-reward ratio
        risk_reward = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')

        # Maximum drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
            max_drawdown = abs(equity_df['drawdown'].min())
        else:
            max_drawdown = 0

        # Collect results
        self.trade_results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'risk_reward': risk_reward,
            'max_drawdown': max_drawdown
        }

        # Log results
        self.logger.info(f"--- Trade Analysis ---")
        self.logger.info(f"Total trades: {total_trades}")
        self.logger.info(f"Win rate: {win_rate:.2f}%")
        self.logger.info(f"Average profit: ${avg_profit:.2f}")
        self.logger.info(f"Average loss: ${avg_loss:.2f}")
        self.logger.info(f"Risk-reward ratio: {risk_reward:.2f}")
        self.logger.info(f"Maximum drawdown: {max_drawdown:.2f}%")

    def plot_results(self, save_path=None):
        # Create equity curve
        equity_df = pd.DataFrame(self.equity_curve)

        if equity_df.empty:
            self.logger.warning("No equity data to plot")
            return

        # Set up the figure
        plt.figure(figsize=(12, 8))

        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(equity_df['date'], equity_df['equity'], label='Equity')
        plt.plot(equity_df['date'], equity_df['balance'], label='Balance', linestyle='--')
        plt.title(f'{self.symbol} Backtest Results')
        plt.ylabel('Account Value ($)')
        plt.grid(True)
        plt.legend()

        # Plot drawdown
        if len(equity_df) > 0:
            plt.subplot(2, 1, 2)
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
            plt.fill_between(equity_df['date'], equity_df['drawdown'], 0, color='red', alpha=0.3)
            plt.plot(equity_df['date'], equity_df['drawdown'], color='red')
            plt.title('Drawdown')
            plt.ylabel('Drawdown (%)')
            plt.xlabel('Date')
            plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Results plot saved to {save_path}")
        else:
            plt.show()

        # Create additional plots for trade analysis
        trades = [t for t in self.broker.trade_history if 'close_price' in t]
        if trades:
            trades_df = pd.DataFrame(trades)

            plt.figure(figsize=(12, 10))

            # Plot cumulative profit
            plt.subplot(3, 1, 1)
            trades_df['cumulative_profit'] = trades_df['profit'].cumsum()
            plt.plot(trades_df['close_time'], trades_df['cumulative_profit'])
            plt.title('Cumulative Profit')
            plt.grid(True)

            # Plot profit distribution
            plt.subplot(3, 1, 2)
            plt.hist(trades_df['profit'], bins=20, color='skyblue', edgecolor='black')
            plt.axvline(0, color='red', linestyle='--')
            plt.title('Profit Distribution')
            plt.xlabel('Profit ($)')
            plt.ylabel('Frequency')
            plt.grid(True)

            # Plot duration distribution
            plt.subplot(3, 1, 3)
            # Convert timedelta to hours
            duration_hours = trades_df['duration'].dt.total_seconds() / 3600
            plt.hist(duration_hours, bins=20, color='lightgreen', edgecolor='black')
            plt.title('Trade Duration Distribution')
            plt.xlabel('Duration (hours)')
            plt.ylabel('Frequency')
            plt.grid(True)

            plt.tight_layout()

            if save_path:
                path_parts = save_path.split('.')
                trade_path = path_parts[0] + '_trades.' + path_parts[1]
                plt.savefig(trade_path)
                self.logger.info(f"Trade analysis plot saved to {trade_path}")
            else:
                plt.show()


def run_backtest(config_dict, years=10, data_dir="historical_data"):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("backtest.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("backtest_runner")

    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    # Define symbol and timeframes
    symbol = config_dict.get('SYMBOL', 'USDJPY')
    timeframes = [
        config_dict.get('PRIMARY_TIMEFRAME', '15min'),
        config_dict.get('CONFIRM_TIMEFRAME', '1H')
    ]

    logger.info(f"Setting up backtest for {symbol} over {years} years")
    logger.info(f"Date range: {start_date} to {end_date}")

    # Initialize backtester
    backtester = Backtester(
        symbol=symbol,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date,
        config=config_dict,
        initial_balance=10000.0
    )

    # Load or generate data
    if not backtester.load_data(data_dir):
        logger.error("Failed to load or generate data")
        return None

    # Run backtest
    def progress_update(progress):
        if int(progress * 100) % 10 == 0:
            logger.info(f"Backtest progress: {progress * 100:.0f}%")

    results = backtester.run(progress_callback=progress_update)

    # Plot results
    backtester.plot_results(save_path=f"backtest_results_{symbol}_{years}yr.png")

    # Generate summary report
    summary = f"""
    ======= BACKTEST SUMMARY =======
    Symbol: {symbol}
    Period: {start_date.date()} to {end_date.date()} ({years} years)

    Initial Balance: ${results['initial_balance']:.2f}
    Final Balance: ${results['final_balance']:.2f}

    Total Profit: ${results['profit']:.2f} ({results['profit_pct']:.2f}%)

    Total Trades: {backtester.trade_results['total_trades']}
    Win Rate: {backtester.trade_results['win_rate']:.2f}%
    Risk-Reward Ratio: {backtester.trade_results['risk_reward']:.2f}

    Max Drawdown: {backtester.trade_results['max_drawdown']:.2f}%
    ==============================
    """

    logger.info(summary)

    with open(f"backtest_report_{symbol}_{years}yr.txt", "w") as f:
        f.write(summary)

    return results


if __name__ == "__main__":
    import config
    import argparse

    parser = argparse.ArgumentParser(description='USDJPY Trading Bot Backtester')
    parser.add_argument('--years', type=int, default=10, help='Number of years to backtest')
    parser.add_argument('--data-dir', type=str, default='historical_data', help='Directory with historical data')
    args = parser.parse_args()

    run_backtest(config.__dict__, years=args.years, data_dir=args.data_dir)
