import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging
import time
import zipfile
import io
import struct


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("data_download.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("data_downloader")


def parse_bi5(data):
    """Parse Dukascopy's bi5 binary format"""
    records = []
    for i in range(0, len(data), 20):  # Each record is 20 bytes
        if i + 20 <= len(data):
            timestamp, open_bid, high_bid, low_bid, close_bid, volume = struct.unpack('>Lfffff', data[i:i + 20])
            # timestamp is milliseconds since epoch
            date = datetime.fromtimestamp(timestamp / 1000.0)
            records.append([date, open_bid, high_bid, low_bid, close_bid, volume])

    return pd.DataFrame(records, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])


def download_dukascopy_data(symbol, start_date, end_date, timeframe, output_dir="historical_data"):
    logger = logging.getLogger("dukascopy_downloader")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Map timeframe to Dukascopy format
    tf_map = {
        "1min": 1,
        "5min": 5,
        "15min": 15,
        "30min": 30,
        "1H": 60,
        "4H": 240,
        "1D": 1440
    }

    if timeframe not in tf_map:
        logger.error(f"Unsupported timeframe: {timeframe}")
        return None

    # Process date range
    current_date = start_date
    all_data = []

    base_url = "https://datafeed.dukascopy.com/datafeed"

    logger.info(f"Downloading {symbol} {timeframe} data from {start_date.date()} to {end_date.date()}")

    # Dukascopy uses USDJPY not USD/JPY
    symbol = symbol.replace("/", "")

    # Loop through each day in the date range
    while current_date <= end_date:
        # Skip weekends (Forex market is closed)
        if current_date.weekday() < 5:  # Monday to Friday
            year = current_date.year
            month = current_date.month - 1  # Dukascopy months are 0-based
            day = current_date.day

            # Construct URL for this day's data
            url = f"{base_url}/{symbol}/{year}/{month:02d}/{day:02d}/BID_{tf_map[timeframe]}.bi5"

            try:
                logger.debug(f"Downloading {url}")
                response = requests.get(url, timeout=30)

                if response.status_code == 200:
                    # Process the data
                    data = response.content

                    if len(data) > 0:
                        try:
                            # For bi5 files, use zipfile to extract
                            with zipfile.ZipFile(io.BytesIO(data)) as z:
                                for filename in z.namelist():
                                    with z.open(filename) as f:
                                        # Read binary data
                                        binary_data = f.read()

                                        # Parse binary format
                                        day_data = parse_bi5(binary_data)
                                        if not day_data.empty:
                                            all_data.append(day_data)
                        except Exception as e:
                            logger.error(f"Error processing data: {e}")
                else:
                    logger.debug(f"No data available for {current_date.date()} (HTTP {response.status_code})")
            except Exception as e:
                logger.error(f"Error downloading data for {current_date.date()}: {e}")

            # Sleep to avoid overwhelming the server
            time.sleep(0.5)

        # Move to next day
        current_date += timedelta(days=1)

        # Show progress every week
        if current_date.weekday() == 0:
            progress = (current_date - start_date).days / (end_date - start_date).days * 100
            logger.info(f"Download progress: {progress:.1f}%")

    if not all_data:
        logger.error("No data was downloaded")
        return None

    # Combine all data
    combined_data = pd.concat(all_data)

    # Sort by datetime
    combined_data = combined_data.sort_values('datetime')

    # Remove duplicates
    combined_data = combined_data.drop_duplicates(subset=['datetime'])

    # Set datetime as index
    combined_data.set_index('datetime', inplace=True)

    # Save to CSV
    output_file = f"{output_dir}/{symbol}_{timeframe}.csv"
    combined_data.to_csv(output_file)

    logger.info(f"Downloaded {len(combined_data)} candles for {symbol} {timeframe}")
    logger.info(f"Data range: {combined_data.index.min()} to {combined_data.index.max()}")
    logger.info(f"Data saved to {output_file}")

    return combined_data


def generate_synthetic_data(symbol, start_date, end_date, timeframe, output_dir="historical_data"):
    """Generate synthetic data when real data cannot be downloaded"""
    logger = logging.getLogger("synthetic_data_generator")

    logger.info(f"Generating synthetic {symbol} {timeframe} data from {start_date.date()} to {end_date.date()}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

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
        logger.error(f"Unsupported timeframe format: {timeframe}")
        return None

    # Generate dates
    current_date = start_date
    dates = []

    while current_date <= end_date:
        # Skip weekends for forex
        if current_date.weekday() < 5:  # Monday to Friday
            # Skip after Friday 22:00 UTC and before Sunday 22:00 UTC
            hour = current_date.hour
            weekday = current_date.weekday()

            if not (weekday == 4 and hour >= 22) and not (weekday == 6 and hour < 22):
                dates.append(current_date)

        # Move to next candle
        current_date += candle_duration

    logger.info(f"Generating {len(dates)} candles")

    # Generate synthetic price data
    np.random.seed(42)  # For reproducibility

    # Set initial price based on symbol
    if symbol == "USDJPY":
        initial_price = 110.0
        drift = 0.00002
        volatility = 0.0008
    elif symbol == "EURUSD":
        initial_price = 1.1500
        drift = 0.00001
        volatility = 0.0005
    elif symbol == "GBPUSD":
        initial_price = 1.3000
        drift = 0.00001
        volatility = 0.0006
    elif symbol == "XAUUSD":  # Gold
        initial_price = 1800.0
        drift = 0.00003
        volatility = 0.001
    else:
        initial_price = 100.0
        drift = 0.00001
        volatility = 0.0005

    # Use random walk with drift for price generation
    returns = np.random.normal(drift, volatility, len(dates))

    # Add some autocorrelation to simulate trending markets
    for i in range(1, len(returns)):
        returns[i] = 0.8 * returns[i] + 0.2 * returns[i - 1]

    # Calculate prices
    log_returns = np.cumsum(returns)
    prices = initial_price * np.exp(log_returns)

    # Generate OHLC data with realistic properties
    data = []
    for i, date in enumerate(dates):
        close_price = prices[i]

        # Calculate reasonable open/high/low based on volatility
        daily_volatility = volatility * initial_price

        # Open price (previous close with small gap)
        if i > 0:
            open_price = prices[i - 1] * (1 + np.random.normal(0, volatility * 0.2))
        else:
            open_price = close_price * (1 + np.random.normal(0, volatility * 0.5))

        # High and low prices (realistic range)
        price_range = daily_volatility * 2.5
        high_price = max(open_price, close_price) + abs(np.random.normal(0, price_range))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, price_range))

        # Volume (higher on big moves)
        price_change = abs(close_price - open_price) / close_price
        volume = np.random.exponential(100) * (1 + price_change * 10)

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

    # Save to CSV
    output_file = f"{output_dir}/{symbol}_{timeframe}.csv"
    df.to_csv(output_file)

    logger.info(f"Generated {len(df)} synthetic candles for {symbol} {timeframe}")
    logger.info(f"Data range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Data saved to {output_file}")

    return df


def download_or_generate_data(symbol, start_date, end_date, timeframes, output_dir="historical_data",
                              use_synthetic=False):
    """Download real data or generate synthetic data for backtesting"""
    logger = setup_logging()

    logger.info(f"Preparing data for {symbol} from {start_date} to {end_date}")

    result = {}

    for timeframe in timeframes:
        logger.info(f"Processing {timeframe} timeframe")

        if use_synthetic:
            logger.info("Using synthetic data generation")
            data = generate_synthetic_data(symbol, start_date, end_date, timeframe, output_dir)
        else:
            try:
                logger.info("Attempting to download real data")
                data = download_dukascopy_data(symbol, start_date, end_date, timeframe, output_dir)

                if data is None or len(data) < 100:
                    logger.warning("Download failed or insufficient data, falling back to synthetic data")
                    data = generate_synthetic_data(symbol, start_date, end_date, timeframe, output_dir)
            except Exception as e:
                logger.error(f"Error downloading data: {e}")
                logger.info("Falling back to synthetic data")
                data = generate_synthetic_data(symbol, start_date, end_date, timeframe, output_dir)

        result[timeframe] = data

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download or generate historical price data for backtesting')
    parser.add_argument('--symbol', type=str, default='USDJPY', help='Trading symbol (e.g., USDJPY, EURUSD)')
    parser.add_argument('--years', type=int, default=10, help='Number of years of data to download')
    parser.add_argument('--output-dir', type=str, default='historical_data', help='Output directory for data files')
    parser.add_argument('--synthetic', action='store_true', help='Generate synthetic data instead of downloading')
    args = parser.parse_args()

    # Setup date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * args.years)

    # Define timeframes to download
    timeframes = ['15min', '1H']

    # Download or generate data
    download_or_generate_data(
        symbol=args.symbol,
        start_date=start_date,
        end_date=end_date,
        timeframes=timeframes,
        output_dir=args.output_dir,
        use_synthetic=args.synthetic
    )