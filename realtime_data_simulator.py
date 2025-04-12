"""
Real-time market data simulator for trading platform
This module provides realistic stock market data in real-time
without requiring historical data APIs like DataBento
"""

import os
import json
import logging
import random
import time
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealtimeDataSimulator:
    """
    Simulator for real-time market data including quotes, trades, and order book
    Uses current date and time for all timestamps
    """
    
    def __init__(self):
        """Initialize the simulator with base stock data"""
        self.stocks = {
            'AAPL': {'price': 190.32, 'volatility': 0.012, 'volume': 15000},
            'MSFT': {'price': 425.22, 'volatility': 0.01, 'volume': 12000},
            'AMZN': {'price': 182.38, 'volatility': 0.015, 'volume': 8000},
            'GOOG': {'price': 178.71, 'volatility': 0.011, 'volume': 9000},
            'META': {'price': 480.42, 'volatility': 0.018, 'volume': 10000},
            'TSLA': {'price': 175.34, 'volatility': 0.025, 'volume': 20000},
            'NVDA': {'price': 880.08, 'volatility': 0.022, 'volume': 18000},
            'JPM': {'price': 184.71, 'volatility': 0.009, 'volume': 7000},
            'BAC': {'price': 37.38, 'volatility': 0.01, 'volume': 6000},
            'V': {'price': 270.92, 'volatility': 0.008, 'volume': 5000},
        }
        
        # Track last update time for each stock to enable realistic price movements
        self.last_update = {symbol: datetime.now() for symbol in self.stocks}
        self.last_trades = {symbol: [] for symbol in self.stocks}
        self.market_mood = 0  # Can range from -1.0 (bearish) to 1.0 (bullish)
        
        # Start a background process to update prices to make them dynamic
        self._update_market_mood()
        logger.info("Realtime data simulator initialized")
    
    def _update_market_mood(self):
        """Update the overall market mood (sentiment)"""
        # Market mood changes slowly over time
        self.market_mood += random.uniform(-0.1, 0.1)
        self.market_mood = max(-1.0, min(1.0, self.market_mood))
    
    def _get_current_price(self, symbol: str) -> float:
        """
        Get the current price for a symbol, updating it based on time elapsed
        since last query to simulate realistic price movements
        """
        if symbol not in self.stocks:
            # If unknown symbol, create it with a random price
            self.stocks[symbol] = {
                'price': random.uniform(50.0, 500.0),
                'volatility': random.uniform(0.01, 0.025),
                'volume': int(random.uniform(5000, 20000))
            }
            self.last_update[symbol] = datetime.now()
            self.last_trades[symbol] = []
        
        # Get time elapsed since last update
        now = datetime.now()
        elapsed = (now - self.last_update[symbol]).total_seconds()
        
        # Only update if time has passed
        if elapsed > 0:
            # Calculate a realistic price move
            stock_data = self.stocks[symbol]
            base_price = stock_data['price']
            volatility = stock_data['volatility']
            
            # Calculate price change, influenced by volatility, market mood,
            # and elapsed time (more time = potentially bigger move)
            market_factor = 1 + (self.market_mood * 0.5)  # Market mood influence
            change_pct = random.normalvariate(0, volatility) * market_factor * min(elapsed, 5)
            
            # Apply change as percentage of current price
            new_price = base_price * (1 + change_pct)
            # Ensure price never goes negative
            new_price = max(0.01, new_price)
            
            # Update the price
            self.stocks[symbol]['price'] = new_price
            self.last_update[symbol] = now
            
            # Record this as a trade if significant enough change
            if abs(change_pct) > 0.0001:
                size = int(random.uniform(50, stock_data['volume'] / 100))
                exchange = random.choice(['NASDAQ', 'NYSE', 'NYSE', 'NASDAQ', 'ARCA'])
                tape = 'C' if exchange == 'NASDAQ' else 'A'
                is_buyer_maker = random.choice([True, False])
                
                trade = {
                    "timestamp": now.isoformat(),
                    "price": new_price,
                    "size": size,
                    "exchange": exchange,
                    "trade_id": f"{symbol}-{int(time.time())}-{random.randint(1000, 9999)}",
                    "tape": tape,
                    "is_buyer_maker": is_buyer_maker
                }
                
                # Add to trade history, keeping last 100 trades
                self.last_trades[symbol].insert(0, trade)
                if len(self.last_trades[symbol]) > 100:
                    self.last_trades[symbol].pop()
        
        return self.stocks[symbol]['price']
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get the current bid/ask quote for a symbol
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            
        Returns:
            Dict: Quote data with bid, ask, and sizes
        """
        # Update the current price first
        current_price = self._get_current_price(symbol)
        
        # Create a realistic spread based on price
        if current_price < 10:
            spread = current_price * 0.002
        elif current_price < 100:
            spread = current_price * 0.001
        else:
            spread = current_price * 0.0005
            
        # Apply minimum spread
        spread = max(0.01, spread)
        
        # Calculate bid and ask
        half_spread = spread / 2
        bid_price = current_price - half_spread
        ask_price = current_price + half_spread
        
        # Get volume from stock data
        volume = self.stocks[symbol]['volume']
        
        # Create realistic order sizes
        bid_size = int(volume * random.uniform(0.001, 0.004))
        ask_size = int(volume * random.uniform(0.001, 0.004))
        
        # Get a timestamp for right now
        timestamp = datetime.now().isoformat()
        
        return {
            "success": True,
            "symbol": symbol,
            "data": {
                "bid_price": round(bid_price, 4),
                "bid_size": bid_size,
                "ask_price": round(ask_price, 4),
                "ask_size": ask_size,
                "spread": round(spread, 4),
                "last_price": round(current_price, 4),
                "volume": volume,
                "timestamp": timestamp,
                # Add additional data for OHLC
                "open_price": round(current_price * random.uniform(0.99, 1.01), 4),  
                "high_price": round(current_price * random.uniform(1.005, 1.02), 4),
                "low_price": round(current_price * random.uniform(0.98, 0.995), 4),
                "close_price": round(current_price, 4),
                "vwap": round(current_price * random.uniform(0.995, 1.005), 4)
            }
        }
    
    def get_level2_data(self, symbol: str, depth: int = 10) -> Dict[str, Any]:
        """
        Get level 2 market depth for a symbol
        
        Args:
            symbol (str): Stock symbol
            depth (int): Number of price levels to include
            
        Returns:
            Dict: Level 2 order book data
        """
        # Update the current price first
        current_price = self._get_current_price(symbol)
        
        # Create appropriate tick size based on price
        if current_price < 1:
            tick_size = 0.0001
        elif current_price < 10:
            tick_size = 0.001
        elif current_price < 100:
            tick_size = 0.01
        else:
            tick_size = 0.1
            
        # Create realistic bid/ask spread
        spread = max(tick_size, current_price * 0.001)
        
        # Calculate inside bid/ask
        base_bid = current_price - (spread / 2)
        base_ask = current_price + (spread / 2)
        
        # Round to appropriate tick size
        base_bid = round(base_bid / tick_size) * tick_size
        base_ask = round(base_ask / tick_size) * tick_size
        
        # Generate bid levels (price descending)
        bid_levels = []
        for i in range(depth):
            price = round(base_bid - (i * tick_size), 4)
            # Larger sizes at whole number and half number price levels
            size_multiplier = 3.0 if price % 1 == 0 else (2.0 if price % 0.5 == 0 else 1.0)
            # Size decays as we move away from the inside market
            size = int(500 * (0.9 ** i) * size_multiplier)
            bid_levels.append({"price": price, "size": size})
            
        # Generate ask levels (price ascending)
        ask_levels = []
        for i in range(depth):
            price = round(base_ask + (i * tick_size), 4)
            # Larger sizes at whole number and half number price levels
            size_multiplier = 3.0 if price % 1 == 0 else (2.0 if price % 0.5 == 0 else 1.0)
            # Size decays as we move away from the inside market
            size = int(500 * (0.9 ** i) * size_multiplier)
            ask_levels.append({"price": price, "size": size})
            
        return {
            "success": True,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "bids": bid_levels,
            "asks": ask_levels
        }
        
    def get_time_and_sales(self, symbol: str, limit: int = 30) -> Dict[str, Any]:
        """
        Get time and sales data (recent trades) for a symbol
        
        Args:
            symbol (str): Stock symbol
            limit (int): Maximum number of trades to return
            
        Returns:
            Dict: Time and sales data
        """
        # Make sure the price is updated
        self._get_current_price(symbol)
        
        # Get existing trades
        trades = self.last_trades.get(symbol, [])
        
        # If we don't have enough trades, generate more
        current_price = self.stocks[symbol]['price']
        base_volume = self.stocks[symbol].get('volume', 10000)
        
        # Generate additional trades if needed
        if len(trades) < limit:
            # Calculate how many more trades we need
            needed_trades = limit - len(trades)
            existing_count = len(trades)
            
            # Generate historical trades going backward in time
            for i in range(needed_trades):
                # Time goes further back with each trade
                # Existing trades are already in reverse chronological order (newest first)
                seconds_back = (existing_count + i + 1) * random.uniform(10, 120)
                trade_time = datetime.now() - timedelta(seconds=seconds_back)
                
                # Price should be close to current but show some variance
                # More recent trades should be closer to current price
                time_factor = 1 + (0.01 * (i / limit))  # Prices diverge more as we go back
                price_variance = current_price * random.uniform(-0.005, 0.005) * time_factor
                trade_price = current_price + price_variance
                
                # Size should be realistic
                size = int(random.uniform(50, base_volume / 100))
                
                # Other trade attributes
                exchange = random.choice(['NASDAQ', 'NYSE', 'NYSE', 'NASDAQ', 'ARCA'])
                tape = 'C' if exchange == 'NASDAQ' else 'A'
                is_buyer_maker = random.choice([True, False])
                
                trade = {
                    "timestamp": trade_time.isoformat(),
                    "price": round(trade_price, 4),
                    "size": size,
                    "exchange": exchange,
                    "trade_id": f"{symbol}-{int(time.time())}-{i}",
                    "tape": tape,
                    "is_buyer_maker": is_buyer_maker
                }
                
                trades.append(trade)
            
            # Update our stored trades
            self.last_trades[symbol] = trades
            
        # Return only the requested number of trades
        return {
            "success": True,
            "symbol": symbol,
            "trades": trades[:limit]
        }
    
    def get_historical_bars(self, symbol: str, interval: str = 'day', limit: int = 30) -> Dict[str, Any]:
        """
        Get historical bar data (OHLCV) for a symbol
        
        Args:
            symbol (str): Stock symbol
            interval (str): Time interval ('minute', 'hour', 'day')
            limit (int): Number of bars to return
            
        Returns:
            Dict: Historical bar data
        """
        # Make sure the price is updated
        current_price = self._get_current_price(symbol)
        
        # Determine time interval in seconds
        if interval == 'minute':
            seconds_per_interval = 60
            volatility_factor = 0.5
        elif interval == 'hour':
            seconds_per_interval = 3600
            volatility_factor = 1.0
        else:  # day
            seconds_per_interval = 86400
            volatility_factor = 2.0
            
        bars = []
        base_volatility = self.stocks[symbol]['volatility'] * volatility_factor
        
        # Generate bars going backward in time
        for i in range(limit):
            # Each bar is further back in time
            bar_end = datetime.now() - timedelta(seconds=i * seconds_per_interval)
            bar_start = bar_end - timedelta(seconds=seconds_per_interval)
            
            # Calculate price variance for this bar
            # Variance increases as we go back in time
            variance_factor = 1 + (i / 20)  # Incremental increase in variance
            
            # Calculate OHLC prices with realistic relationships
            price_range = current_price * base_volatility * variance_factor
            
            # Randomly choose if this bar is up or down
            is_up = random.random() > 0.5
            
            if is_up:
                close_price = current_price * random.uniform(1, 1 + (base_volatility * variance_factor))
                open_price = close_price * random.uniform(1 - (base_volatility * 0.7), 1)
                high_price = close_price * random.uniform(1, 1 + (base_volatility * 0.3))
                low_price = open_price * random.uniform(1 - (base_volatility * 0.5), 1)
            else:
                close_price = current_price * random.uniform(1 - (base_volatility * variance_factor), 1)
                open_price = close_price * random.uniform(1, 1 + (base_volatility * 0.7))
                high_price = open_price * random.uniform(1, 1 + (base_volatility * 0.3))
                low_price = close_price * random.uniform(1 - (base_volatility * 0.5), 1)
            
            # Ensure high is highest and low is lowest
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Volume should be realistic and vary by bar
            volume = int(self.stocks[symbol]['volume'] * random.uniform(0.5, 1.5))
            
            # VWAP should be between low and high, weighted toward close
            vwap = (close_price * 0.6) + (open_price * 0.2) + (high_price * 0.1) + (low_price * 0.1)
            
            bar = {
                "timestamp": bar_end.isoformat(),
                "open": round(open_price, 4),
                "high": round(high_price, 4),
                "low": round(low_price, 4),
                "close": round(close_price, 4),
                "volume": volume,
                "vwap": round(vwap, 4)
            }
            
            bars.append(bar)
            
            # Update current price for next iteration to create realistic price series
            current_price = close_price
            
        return {
            "success": True,
            "symbol": symbol,
            "bars": bars
        }


# Create a singleton instance
realtime_simulator = RealtimeDataSimulator()