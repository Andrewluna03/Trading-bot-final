"""
Interactive Brokers integration for trading application
Provides real-time market data and trading execution capabilities
"""

import os
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

# Import ib_insync library for Interactive Brokers API
from ib_insync import IB, Contract, Stock, Forex, Option, Future, Crypto, TagValue
import nest_asyncio

# Apply nest_asyncio to allow asyncio to work in Jupyter-like environments
nest_asyncio.apply()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IBClient:
    """
    Client for Interactive Brokers API to fetch real-time market data
    and execute trades
    """
    
    def __init__(self, connect_on_init=False):
        """
        Initialize the Interactive Brokers client
        
        Args:
            connect_on_init (bool): Whether to connect upon initialization
        """
        self.ib = IB()
        self.connected = False
        self.market_data_subscriptions = {}
        
        # If connect_on_init is True, try to connect
        if connect_on_init:
            self.connect()
    
    def connect(self, host='127.0.0.1', port=7497, client_id=1, read_only=True):
        """
        Connect to Interactive Brokers TWS or Gateway
        
        Args:
            host (str): Host address (default: 127.0.0.1)
            port (int): Port (7497 for TWS, 4002 for Gateway)
            client_id (int): Client ID (must be unique)
            read_only (bool): Whether to connect in read-only mode
            
        Returns:
            bool: True if connected successfully, False otherwise
        """
        try:
            self.ib.connect(host, port, clientId=client_id, readonly=read_only)
            self.connected = self.ib.isConnected()
            
            if self.connected:
                logger.info(f"Connected to Interactive Brokers on {host}:{port}")
                return True
            else:
                logger.error("Failed to connect to Interactive Brokers")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Interactive Brokers: {str(e)}")
            self.connected = False
            return False
    
    def disconnect(self):
        """
        Disconnect from Interactive Brokers
        
        Returns:
            bool: True if disconnected successfully
        """
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from Interactive Brokers")
        return True
    
    def is_connected(self):
        """
        Check if connected to Interactive Brokers
        
        Returns:
            bool: True if connected
        """
        return self.ib.isConnected()
    
    def create_stock_contract(self, symbol: str, exchange: str = 'SMART', currency: str = 'USD'):
        """
        Create a stock contract for the given symbol
        
        Args:
            symbol (str): Stock symbol
            exchange (str): Exchange (default: SMART)
            currency (str): Currency (default: USD)
            
        Returns:
            Contract: Stock contract
        """
        return Stock(symbol, exchange, currency)
    
    def create_forex_contract(self, pair: str, exchange: str = 'IDEALPRO', currency: str = None):
        """
        Create a forex contract for the given currency pair
        
        Args:
            pair (str): Currency pair (e.g., 'EURUSD')
            exchange (str): Exchange (default: IDEALPRO)
            currency (str): Currency (default: None)
            
        Returns:
            Contract: Forex contract
        """
        if len(pair) == 6:
            base_currency = pair[:3]
            quote_currency = pair[3:]
            return Forex(base_currency, quote_currency, exchange)
        else:
            logger.error(f"Invalid currency pair format: {pair}. Expected format: 'EURUSD'")
            return None
    
    def create_option_contract(self, symbol: str, expiry: str, strike: float, 
                               right: str, exchange: str = 'SMART', currency: str = 'USD'):
        """
        Create an option contract
        
        Args:
            symbol (str): Underlying symbol
            expiry (str): Expiration date in 'YYYYMMDD' format
            strike (float): Strike price
            right (str): Option right ('C' for Call, 'P' for Put)
            exchange (str): Exchange (default: SMART)
            currency (str): Currency (default: USD)
            
        Returns:
            Contract: Option contract
        """
        return Option(symbol, expiry, strike, right, exchange, currency)
    
    def create_future_contract(self, symbol: str, expiry: str, exchange: str, currency: str = 'USD'):
        """
        Create a future contract
        
        Args:
            symbol (str): Future symbol (e.g., 'ES')
            expiry (str): Expiration date in 'YYYYMMDD' format
            exchange (str): Exchange (e.g., 'CME')
            currency (str): Currency (default: USD)
            
        Returns:
            Contract: Future contract
        """
        return Future(symbol, expiry, exchange, currency)
    
    def create_crypto_contract(self, symbol: str, currency: str = 'USD', exchange: str = 'PAXOS'):
        """
        Create a cryptocurrency contract
        
        Args:
            symbol (str): Crypto symbol (e.g., 'BTC')
            currency (str): Currency (default: USD)
            exchange (str): Exchange (default: PAXOS)
            
        Returns:
            Contract: Crypto contract
        """
        return Crypto(symbol, currency, exchange)
    
    def get_market_data(self, symbol: str, security_type: str = 'STK', exchange: str = 'SMART', 
                        currency: str = 'USD', **kwargs) -> Dict[str, Any]:
        """
        Get real-time market data for a symbol
        
        Args:
            symbol (str): Symbol to fetch data for
            security_type (str): Security type (STK, CASH, OPT, FUT, CRYPTO)
            exchange (str): Exchange (default: SMART)
            currency (str): Currency (default: USD)
            **kwargs: Additional parameters for specific security types
            
        Returns:
            Dict: Market data response or error
        """
        try:
            if not self.connected:
                logger.warning("Not connected to Interactive Brokers. Attempting to connect...")
                if not self.connect():
                    return {
                        "success": False,
                        "error": "Not connected to Interactive Brokers",
                        "symbol": symbol
                    }
            
            # Create contract based on security type
            contract = None
            if security_type == 'STK':
                contract = self.create_stock_contract(symbol, exchange, currency)
            elif security_type == 'CASH':
                contract = self.create_forex_contract(symbol, 'IDEALPRO')
            elif security_type == 'OPT':
                expiry = kwargs.get('expiry', '')
                strike = kwargs.get('strike', 0.0)
                right = kwargs.get('right', 'C')
                contract = self.create_option_contract(symbol, expiry, strike, right, exchange, currency)
            elif security_type == 'FUT':
                expiry = kwargs.get('expiry', '')
                contract = self.create_future_contract(symbol, expiry, exchange, currency)
            elif security_type == 'CRYPTO':
                contract = self.create_crypto_contract(symbol, currency, exchange)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported security type: {security_type}",
                    "symbol": symbol
                }
            
            if not contract:
                return {
                    "success": False,
                    "error": "Failed to create contract",
                    "symbol": symbol
                }
            
            # Request market data
            ticker = self.ib.reqMktData(contract)
            
            # Wait for market data to populate
            timeout = 5  # seconds
            start_time = time.time()
            while time.time() - start_time < timeout:
                if ticker.last or ticker.bid or ticker.ask:
                    break
                self.ib.sleep(0.1)
            
            # Store subscription for later cancellation
            self.market_data_subscriptions[symbol] = contract
            
            # Format market data
            market_data = {
                "success": True,
                "symbol": symbol,
                "data": {
                    "last_price": ticker.last if ticker.last else ticker.close,
                    "bid_price": ticker.bid,
                    "bid_size": ticker.bidSize,
                    "ask_price": ticker.ask,
                    "ask_size": ticker.askSize,
                    "volume": ticker.volume,
                    "open_price": ticker.open,
                    "high_price": ticker.high,
                    "low_price": ticker.low,
                    "close_price": ticker.close,
                    "vwap": ticker.vwap,
                    "timestamp": datetime.now().isoformat(),
                }
            }
            
            # Calculate spread
            if ticker.bid and ticker.ask:
                market_data["data"]["spread"] = round(ticker.ask - ticker.bid, 4)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to get market data: {str(e)}",
                "symbol": symbol
            }
    
    def get_quotes(self, symbol: str, security_type: str = 'STK', exchange: str = 'SMART',
                  currency: str = 'USD', **kwargs) -> Dict[str, Any]:
        """
        Get current NBBO (National Best Bid and Offer) quotes for a symbol
        
        Args:
            symbol (str): Symbol to fetch data for
            security_type (str): Security type (STK, CASH, OPT, FUT, CRYPTO)
            exchange (str): Exchange (default: SMART)
            currency (str): Currency (default: USD)
            **kwargs: Additional parameters for specific security types
            
        Returns:
            Dict: Quote data response or error
        """
        # This is essentially the same as get_market_data but focused on quotes
        return self.get_market_data(symbol, security_type, exchange, currency, **kwargs)
    
    def get_order_book(self, symbol: str, security_type: str = 'STK', exchange: str = 'SMART',
                      currency: str = 'USD', depth: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Get order book (market depth) for a symbol
        
        Args:
            symbol (str): Symbol to fetch data for
            security_type (str): Security type (STK, CASH, OPT, FUT, CRYPTO)
            exchange (str): Exchange (default: SMART)
            currency (str): Currency (default: USD)
            depth (int): Depth of order book (default: 10)
            **kwargs: Additional parameters for specific security types
            
        Returns:
            Dict: Order book data response or error
        """
        try:
            if not self.connected:
                logger.warning("Not connected to Interactive Brokers. Attempting to connect...")
                if not self.connect():
                    return {
                        "success": False,
                        "error": "Not connected to Interactive Brokers",
                        "symbol": symbol
                    }
            
            # Create contract based on security type
            contract = None
            if security_type == 'STK':
                contract = self.create_stock_contract(symbol, exchange, currency)
            elif security_type == 'CASH':
                contract = self.create_forex_contract(symbol, 'IDEALPRO')
            elif security_type == 'OPT':
                expiry = kwargs.get('expiry', '')
                strike = kwargs.get('strike', 0.0)
                right = kwargs.get('right', 'C')
                contract = self.create_option_contract(symbol, expiry, strike, right, exchange, currency)
            elif security_type == 'FUT':
                expiry = kwargs.get('expiry', '')
                contract = self.create_future_contract(symbol, expiry, exchange, currency)
            elif security_type == 'CRYPTO':
                contract = self.create_crypto_contract(symbol, currency, exchange)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported security type: {security_type}",
                    "symbol": symbol
                }
            
            if not contract:
                return {
                    "success": False,
                    "error": "Failed to create contract",
                    "symbol": symbol
                }
            
            # Request market depth data
            self.ib.reqMktDepth(contract, depth, False, [])
            
            # Wait for market depth data to populate
            timeout = 5  # seconds
            self.ib.sleep(timeout)
            
            # Get order book data
            domTicks = self.ib.domTicks()
            
            # Format order book data
            bids = []
            asks = []
            
            for tick in domTicks:
                if tick.contract.symbol == symbol:
                    if tick.side == 0:  # Bid
                        bids.append({
                            "price": tick.price,
                            "size": tick.size
                        })
                    elif tick.side == 1:  # Ask
                        asks.append({
                            "price": tick.price,
                            "size": tick.size
                        })
            
            # Sort order book by price (bids descending, asks ascending)
            bids = sorted(bids, key=lambda x: x['price'], reverse=True)
            asks = sorted(asks, key=lambda x: x['price'])
            
            # Limit to requested depth
            bids = bids[:depth]
            asks = asks[:depth]
            
            return {
                "success": True,
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "bids": bids,
                "asks": asks
            }
            
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to get order book data: {str(e)}",
                "symbol": symbol
            }
    
    def get_historical_data(self, symbol: str, duration: str = '1 D', bar_size: str = '1 min',
                           security_type: str = 'STK', exchange: str = 'SMART', currency: str = 'USD',
                           **kwargs) -> Dict[str, Any]:
        """
        Get historical price data for a symbol
        
        Args:
            symbol (str): Symbol to fetch data for
            duration (str): Duration (e.g., '1 D', '1 W', '1 M', '1 Y')
            bar_size (str): Bar size (e.g., '1 min', '5 mins', '1 hour', '1 day')
            security_type (str): Security type (STK, CASH, OPT, FUT, CRYPTO)
            exchange (str): Exchange (default: SMART)
            currency (str): Currency (default: USD)
            **kwargs: Additional parameters for specific security types
            
        Returns:
            Dict: Historical data response or error
        """
        try:
            if not self.connected:
                logger.warning("Not connected to Interactive Brokers. Attempting to connect...")
                if not self.connect():
                    return {
                        "success": False,
                        "error": "Not connected to Interactive Brokers",
                        "symbol": symbol
                    }
            
            # Create contract based on security type
            contract = None
            if security_type == 'STK':
                contract = self.create_stock_contract(symbol, exchange, currency)
            elif security_type == 'CASH':
                contract = self.create_forex_contract(symbol, 'IDEALPRO')
            elif security_type == 'OPT':
                expiry = kwargs.get('expiry', '')
                strike = kwargs.get('strike', 0.0)
                right = kwargs.get('right', 'C')
                contract = self.create_option_contract(symbol, expiry, strike, right, exchange, currency)
            elif security_type == 'FUT':
                expiry = kwargs.get('expiry', '')
                contract = self.create_future_contract(symbol, expiry, exchange, currency)
            elif security_type == 'CRYPTO':
                contract = self.create_crypto_contract(symbol, currency, exchange)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported security type: {security_type}",
                    "symbol": symbol
                }
            
            if not contract:
                return {
                    "success": False,
                    "error": "Failed to create contract",
                    "symbol": symbol
                }
            
            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',  # Current time
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True
            )
            
            # Format historical data
            bar_data = []
            for bar in bars:
                bar_data.append({
                    "timestamp": bar.date.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "wap": bar.wap,
                    "count": bar.barCount
                })
            
            return {
                "success": True,
                "symbol": symbol,
                "bars": bar_data,
                "bar_count": len(bar_data)
            }
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to get historical data: {str(e)}",
                "symbol": symbol
            }
    
    def cancel_market_data(self, symbol: str = None):
        """
        Cancel market data subscription for a symbol or all symbols
        
        Args:
            symbol (str, optional): Symbol to cancel subscription for.
                                   If None, cancel all subscriptions.
        
        Returns:
            bool: True if successful
        """
        try:
            if symbol:
                if symbol in self.market_data_subscriptions:
                    contract = self.market_data_subscriptions[symbol]
                    self.ib.cancelMktData(contract)
                    del self.market_data_subscriptions[symbol]
                    logger.info(f"Cancelled market data subscription for {symbol}")
            else:
                # Cancel all subscriptions
                for sym, contract in self.market_data_subscriptions.items():
                    self.ib.cancelMktData(contract)
                    logger.info(f"Cancelled market data subscription for {sym}")
                self.market_data_subscriptions = {}
            
            return True
        except Exception as e:
            logger.error(f"Error cancelling market data: {str(e)}")
            return False

# Create a singleton instance
ib_client = IBClient(connect_on_init=False)