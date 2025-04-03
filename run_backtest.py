import argparse
import configparser
import os
import logging
from datetime import datetime, timedelta

from backtester import run_backtest
from download_data import download_or_generate_data
import config


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("backtest_run.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("backtest_runner")


def main():
    logger = setup_logging()

    parser = argparse.ArgumentParser(description='Run backtesting for USDJPY Trading Bot')
    parser.add_argument('--years', type=int, default=10, help='Number of years to backtest')
    parser.add_argument('--data-dir', type=str, default='historical_data', help='Directory with historical data')
    parser.add_argument('--symbol', type=str, default='USDJPY', help='Symbol to backtest')
    parser.add_argument('--download', action='store_true', help='Download new data before backtesting')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data instead of downloading')
    parser.add_argument('--config', type=str, default='secret.ini', help='Path to configuration file')
    args = parser.parse_args()

    logger.info(f"Starting backtest for {args.symbol} over {args.years} years")

    # Load configuration
    if os.path.exists(args.config):
        conf = configparser.ConfigParser()
        conf.read(args.config)

        # Override config settings if needed
        if conf.has_section('Trading'):
            if conf.has_option('Trading', 'risk_percent'):
                risk_percent = conf.getfloat('Trading', 'risk_percent')
                config.RISK_PERCENT = risk_percent
                logger.info(f"Using risk percent from config: {risk_percent}%")

            if conf.has_option('Trading', 'max_open_trades'):
                max_trades = conf.getint('Trading', 'max_open_trades')
                config.MAX_OPEN_TRADES = max_trades
                logger.info(f"Using max open trades from config: {max_trades}")

        # Override symbol from config if provided and not in command line
        if conf.has_section('Broker') and not args.symbol:
            if conf.has_option('Broker', 'symbol'):
                args.symbol = conf.get('Broker', 'symbol')
                logger.info(f"Using symbol from config: {args.symbol}")

    # Update config with symbol
    config.SYMBOL = args.symbol

    # Download data if requested
    if args.download or args.synthetic:
        logger.info("Downloading or generating historical data")

        # Setup date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * args.years)

        # Define timeframes to download
        timeframes = [config.PRIMARY_TIMEFRAME, config.CONFIRM_TIMEFRAME]

        # Download or generate data
        download_or_generate_data(
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date,
            timeframes=timeframes,
            output_dir=args.data_dir,
            use_synthetic=args.synthetic
        )

    # Run backtest
    result = run_backtest(
        config_dict=config.__dict__,
        years=args.years,
        data_dir=args.data_dir
    )

    if result:
        logger.info(f"Backtest completed successfully. Profit: ${result['profit']:.2f} ({result['profit_pct']:.2f}%)")
    else:
        logger.error("Backtest failed")


if __name__ == "__main__":
    main()