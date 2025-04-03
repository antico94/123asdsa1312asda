import time
import logging
import argparse
import configparser
import os
from broker_interface import BrokerInterface
from strategy import TradingStrategy
from position_manager import PositionManager
import config


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("trading_bot.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("trading_bot")


def load_config():
    parser = argparse.ArgumentParser(description='USDJPY Trading Bot')
    parser.add_argument('--config', type=str, default='secret.ini', help='Path to configuration file')
    parser.add_argument('--account', type=str, help='Override broker account ID')
    parser.add_argument('--api-key', type=str, help='Override broker API key')
    parser.add_argument('--symbol', type=str, help='Override trading symbol')
    parser.add_argument('--test', action='store_true', help='Override to run in test mode')
    args = parser.parse_args()

    # Load configuration from file
    if not os.path.exists(args.config):
        raise FileNotFoundError(
            f"Configuration file {args.config} not found. Please create it from secret_template.ini")

    conf = configparser.ConfigParser()
    conf.read(args.config)

    # Create settings dictionary with defaults
    settings = {
        'account_id': conf.get('Broker', 'account_id', fallback='demo1234'),
        'api_key': conf.get('Broker', 'api_key', fallback=''),
        'symbol': conf.get('Broker', 'symbol', fallback='USDJPY'),
        'test_mode': conf.getboolean('Trading', 'test_mode', fallback=True),
        'risk_percent': conf.getfloat('Trading', 'risk_percent', fallback=1.0),
        'max_open_trades': conf.getint('Trading', 'max_open_trades', fallback=2)
    }

    # Override with command-line arguments if provided
    if args.account:
        settings['account_id'] = args.account
    if args.api_key:
        settings['api_key'] = args.api_key
    if args.symbol:
        settings['symbol'] = args.symbol
    if args.test:
        settings['test_mode'] = True

    return settings


def main():
    logger = setup_logging()
    settings = load_config()

    logger.info(f"Starting USDJPY Trading Bot for {settings['symbol']}")
    logger.info(f"Account: {settings['account_id']}, Test Mode: {settings['test_mode']}")

    # Initialize broker interface
    broker = BrokerInterface(
        account_id=settings['account_id'],
        api_key=settings['api_key'],
        symbol=settings['symbol']
    )

    # Update config with settings from ini file
    trading_config = config.__dict__.copy()
    trading_config['RISK_PERCENT'] = settings['risk_percent']
    trading_config['MAX_OPEN_TRADES'] = settings['max_open_trades']
    trading_config['SYMBOL'] = settings['symbol']

    # Initialize strategy and position manager
    strategy = TradingStrategy(broker, trading_config)
    position_manager = PositionManager(broker, trading_config)

    # Trading loop
    try:
        while True:
            # Update position manager
            position_manager.update()

            # Check if we can open a new position
            if position_manager.can_open_new_position():
                # Check for trading signals
                signal = strategy.should_open_position()

                if signal:
                    # Confirm entry
                    if strategy.get_entry_confirmation(signal):
                        logger.info(f"Confirmed {signal} signal, opening position")
                        position_manager.open_position(signal)
                    else:
                        logger.info(f"Signal {signal} detected but confirmation failed")

            # Sleep to avoid excessive CPU usage
            # In a real implementation, you might want to use a timer or event-based approach
            time.sleep(60)  # Check every minute

            if settings['test_mode']:
                # In test mode, we use a simulated price change
                current_bid = broker.get_current_price()['bid']
                current_ask = broker.get_current_price()['ask']

                # Simulate price movement (random walk with drift)
                import random
                drift = random.uniform(-0.05, 0.05)
                new_bid = current_bid + drift
                new_ask = current_bid + drift + 0.02  # Maintain spread

                broker.update_simulated_price(new_bid, new_ask)

                # Print status in test mode
                summary = position_manager.get_position_summary()
                logger.info(
                    f"Test mode - Price: {new_bid:.3f}/{new_ask:.3f}, Positions: {summary['total_positions']}, Profit: {summary['total_profit']:.2f}")

    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        # Close all positions when stopping
        position_manager.close_all_positions("Bot shutdown")
    except Exception as e:
        logger.exception(f"Error in main loop: {e}")
        # Try to close positions on error
        try:
            position_manager.close_all_positions("Error shutdown")
        except:
            logger.exception("Failed to close positions on error")
    finally:
        logger.info("Trading bot shutdown complete")


if __name__ == "__main__":
    main()