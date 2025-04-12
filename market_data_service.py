import os
import logging
import json
import time
import random
from datetime import datetime, timedelta
from dateutil import parser
from typing import Dict, Any, List, Optional, Tuple

from app import db
from models import MarketData, TradeData
from polygon_client import PolygonClient
from databento_client import databento_client
from realtime_data_simulator import realtime_simulator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataService:
    """
    Service to fetch, process and store real-time market data
    """
    
    def __init__(self):
        """Initialize the market data service"""
        self.polygon = PolygonClient()
        self.databento = databento_client
        self.active_symbols = set()  # Set of symbols being actively monitored
        self.use_databento = os.environ.get("DATABENTO_API_KEY") is not None  # Flag to determine data source
        
    def add_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Add a symbol to the active monitoring list
        
        Args:
            symbol (str): Stock symbol to monitor (e.g., AAPL)
            
        Returns:
            Dict: Response with success status and initial data
        """
        symbol = symbol.upper().strip()
        
        if not symbol:
            return {"success": False, "error": "Invalid symbol"}
        
        # Add to active symbols set
        self.active_symbols.add(symbol)
        logger.info(f"Added {symbol} to active monitoring")
        
        # Fetch initial data
        return self.get_market_data(symbol)
    
    def remove_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Remove a symbol from the active monitoring list
        
        Args:
            symbol (str): Stock symbol to stop monitoring
            
        Returns:
            Dict: Response with success status
        """
        symbol = symbol.upper().strip()
        
        if symbol in self.active_symbols:
            self.active_symbols.remove(symbol)
            logger.info(f"Removed {symbol} from active monitoring")
            return {"success": True, "message": f"Stopped monitoring {symbol}"}
        else:
            return {"success": False, "error": f"{symbol} is not being monitored"}
    
    def get_active_symbols(self) -> List[str]:
        """Get the list of currently active symbols"""
        return list(self.active_symbols)
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch and store current market data for a symbol
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict: Market data response or error
        """
        symbol = symbol.upper().strip()
        
        try:
            # Get real-time market data from our simulator
            quote_response = realtime_simulator.get_quote(symbol)
            
            if not quote_response.get('success', False):
                logger.error(f"Error fetching quotes for {symbol}: {quote_response.get('message', 'Unknown error')}")
                return quote_response
            
            if 'data' in quote_response:
                # Extract data from the simulator response
                quote_data = quote_response['data']
                
                # Create a MarketData object
                market_date = datetime.now()
                
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=market_date,
                    bid_price=quote_data.get('bid_price'),
                    bid_size=quote_data.get('bid_size'),
                    ask_price=quote_data.get('ask_price'),
                    ask_size=quote_data.get('ask_size'),
                    last_price=quote_data.get('last_price'),
                    volume=quote_data.get('volume'),
                    vwap=quote_data.get('vwap'),
                    open_price=quote_data.get('open_price'),
                    high_price=quote_data.get('high_price'),
                    low_price=quote_data.get('low_price'),
                    close_price=quote_data.get('close_price'),
                    spread=quote_data.get('spread')
                )
                
                # Store in database
                if market_data:
                    self._save_market_data(market_data)
                
                # Return data
                return {
                    "success": True,
                    "symbol": symbol,
                    "market_data": market_data.to_dict() if market_data else None
                }
            else:
                logger.error(f"No data found in quote response for {symbol}")
                return {
                    "success": False,
                    "error": "No market data available",
                    "symbol": symbol
                }
            
        except Exception as e:
            logger.error(f"Error in get_market_data for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to get market data: {str(e)}",
                "symbol": symbol
            }
    
    def get_time_and_sales(self, symbol: str, limit: int = 50) -> Dict[str, Any]:
        """
        Fetch Time & Sales data (recent trades) for a symbol
        
        Args:
            symbol (str): Stock symbol
            limit (int): Number of recent trades to fetch
            
        Returns:
            Dict: Time & Sales data response or error
        """
        symbol = symbol.upper().strip()
        
        try:
            # Get real-time trade data from our simulator
            trades_response = realtime_simulator.get_time_and_sales(symbol, limit)
            
            if not trades_response.get('success', False):
                logger.error(f"Error fetching time & sales for {symbol}: {trades_response.get('message', 'Unknown error')}")
                return trades_response
            
            # Store trades in database (optional)
            if 'trades' in trades_response and trades_response['trades']:
                for trade_data in trades_response['trades']:
                    try:
                        # Create a TradeData object
                        trade_time = parser.parse(trade_data['timestamp'])
                        
                        trade = TradeData(
                            symbol=symbol,
                            timestamp=trade_time,
                            price=trade_data.get('price'),
                            size=trade_data.get('size'),
                            exchange=trade_data.get('exchange'),
                            trade_id=trade_data.get('trade_id'),
                            tape=trade_data.get('tape'),
                            is_buyer_maker=trade_data.get('is_buyer_maker', False)
                        )
                        
                        # Save to database
                        self._save_trade_data(trade)
                    except Exception as e:
                        logger.warning(f"Error processing trade data: {str(e)}")
            
            # Return trades directly from simulator
            return trades_response
            
        except Exception as e:
            logger.error(f"Error in get_time_and_sales for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to get time & sales data: {str(e)}",
                "symbol": symbol
            }
    
    def get_order_book(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch Level 2 order book data for a symbol
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict: Order book data response or error
        """
        symbol = symbol.upper().strip()
        
        try:
            # Get real-time order book data from our simulator
            l2_response = realtime_simulator.get_level2_data(symbol)
            
            if not l2_response.get('success', False):
                logger.error(f"Error fetching level 2 data for {symbol}: {l2_response.get('message', 'Unknown error')}")
                return l2_response
            
            # Return the data from the simulator directly (already formatted correctly)
            return l2_response
            
        except Exception as e:
            logger.error(f"Error in get_order_book for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to get order book data: {str(e)}",
                "symbol": symbol
            }
    
    def update_all_symbols(self) -> Dict[str, Any]:
        """
        Update market data for all actively monitored symbols
        
        Returns:
            Dict: Results of all updates
        """
        results = {}
        for symbol in self.active_symbols:
            results[symbol] = self.get_market_data(symbol)
            
        return {
            "success": True,
            "symbols_updated": len(results),
            "results": results
        }
    
    def get_analysis(self, symbol: str, use_ai: bool = True, analysis_depth: str = 'standard') -> Dict[str, Any]:
        """
        Generate trading analysis based on recent market data
        
        Args:
            symbol (str): Stock symbol
            use_ai (bool): Whether to use OpenAI for enhanced analysis
            analysis_depth (str): Level of analysis detail (basic, standard, detailed)
            
        Returns:
            Dict: Analysis results
        """
        symbol = symbol.upper().strip()
        
        try:
            # Get recent market data
            market_data = self._get_latest_market_data(symbol)
            if not market_data:
                return {
                    "success": False,
                    "error": f"No recent market data found for {symbol}",
                    "symbol": symbol
                }
            
            # Get recent trades
            recent_trades = self._get_recent_trades(symbol, 100)
            
            # Get order book (Level 2) data
            order_book = None
            try:
                order_book_response = self.get_order_book(symbol)
                if order_book_response.get('success', False):
                    order_book = {
                        'bids': order_book_response.get('bids', []),
                        'asks': order_book_response.get('asks', [])
                    }
            except Exception as e:
                logger.warning(f"Could not fetch order book for {symbol}: {str(e)}")
            
            # Format recent trades for analysis
            formatted_trades = []
            for trade in recent_trades:
                formatted_trades.append(trade.to_dict())
            
            # If AI analysis is requested and we have market data
            if use_ai and market_data:
                # Need to import here to avoid circular imports
                from openai_analyzer import analyze_market_data
                
                # Get AI analysis
                ai_analysis = analyze_market_data(
                    symbol=symbol,
                    market_data=market_data.to_dict(),
                    time_and_sales=formatted_trades,
                    order_book=order_book,
                    analysis_depth=analysis_depth
                )
                
                if ai_analysis.get('success', False):
                    logger.info(f"Successfully got AI analysis for {symbol}")
                    return ai_analysis
                else:
                    logger.warning(f"AI analysis failed for {symbol}, falling back to standard analysis: {ai_analysis.get('error', 'Unknown error')}")
            
            # Standard algorithmic analysis (fallback or if AI not requested)
            # Calculate buying/selling pressure
            buying_pressure, selling_pressure = self._calculate_pressure(recent_trades)
            
            # Calculate trade speed
            trades_per_minute = self._calculate_trade_speed(recent_trades)
            
            # Determine price trend
            price_trend = self._determine_price_trend(recent_trades)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                market_data, 
                buying_pressure, 
                selling_pressure, 
                trades_per_minute, 
                price_trend
            )
            
            # Prepare analysis response
            analysis = {
                "success": True,
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "current_price": market_data.last_price,
                "bid_price": market_data.bid_price,
                "bid_size": market_data.bid_size,
                "ask_price": market_data.ask_price,
                "ask_size": market_data.ask_size,
                "spread": market_data.spread,
                "volume": market_data.volume,
                "buying_pressure": buying_pressure,
                "selling_pressure": selling_pressure,
                "trades_per_minute": trades_per_minute,
                "price_trend": price_trend,
                "recommendation": recommendation,
                "recent_trade_count": len(recent_trades),
                "analysis": f"Algorithmic analysis for {symbol}: {price_trend} trend with {buying_pressure}% buying pressure and {selling_pressure}% selling pressure.",
                "ai_analysis": False
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in get_analysis for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to generate analysis: {str(e)}",
                "symbol": symbol
            }
    
    def _process_databento_quote(self, symbol: str, quote_response: Dict[str, Any]) -> Optional[MarketData]:
        """
        Process quote data from DataBento API into MarketData object
        
        Args:
            symbol (str): Stock symbol
            quote_response (Dict): Response from DataBento NBBO API
            
        Returns:
            Optional[MarketData]: Processed MarketData object or None if error
        """
        try:
            # For development purposes, we'll return simulated market data based on the symbol
            # Get symbol-specific test data from the DataBento client
            test_data = self.databento._get_symbol_test_data(symbol)
            bid_price = test_data['bid_price']
            ask_price = test_data['ask_price']
            last_price = round((bid_price + ask_price) / 2, 2)
            price_base = test_data['price_base']
            
            # Use April 2023 date for timestamp instead of future date (2025)
            market_date = datetime.now() + timedelta(seconds=random.randint(0, 3600))
            
            return MarketData(
                symbol=symbol,
                timestamp=market_date,
                bid_price=bid_price,
                bid_size=test_data['bid_size'],
                ask_price=ask_price,
                ask_size=test_data['ask_size'],
                last_price=last_price,
                volume=test_data['volume'],
                vwap=round(price_base * 0.998, 2),
                open_price=round(price_base * 0.995, 2),
                high_price=round(price_base * 1.01, 2),
                low_price=round(price_base * 0.99, 2),
                close_price=last_price,
                spread=round(ask_price - bid_price, 4)
            )
            
            # In production, we would use the following code:
            # Check if we have the required data in the response
            if 'bidPrice' not in quote_response or 'askPrice' not in quote_response:
                logger.warning(f"Missing required quote data in DataBento response for {symbol}")
                return None
            
            # Get the quote data
            bid_price = quote_response.get('bidPrice', 0)
            bid_size = quote_response.get('bidSize', 0)
            ask_price = quote_response.get('askPrice', 0) 
            ask_size = quote_response.get('askSize', 0)
            
            # Also try to get historical bar data for other fields
            bars_response = self.databento.get_historical_bars(symbol, interval='day', limit=1)
            
            # Default values for additional fields
            open_price = close_price = high_price = low_price = last_price = bid_price
            volume = vwap = 0
            
            if bars_response.get('success', False) and 'bars' in bars_response and bars_response['bars']:
                bar = bars_response['bars'][0]
                open_price = bar.get('open', bid_price)
                close_price = bar.get('close', bid_price)
                high_price = bar.get('high', ask_price)
                low_price = bar.get('low', bid_price)
                volume = bar.get('volume', bid_size + ask_size)
                last_price = close_price
            
            # Create new MarketData object with realistic time
            # Use April 2023 date for timestamp instead of future date (2025)
            market_date = datetime.now() + timedelta(seconds=random.randint(0, 3600))
            
            market_data = MarketData(
                symbol=symbol,
                timestamp=market_date,
                bid_price=bid_price,
                bid_size=bid_size,
                ask_price=ask_price,
                ask_size=ask_size, 
                last_price=last_price,
                volume=volume,
                vwap=vwap,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price
            )
            
            # Calculate spread if both bid and ask are available
            if market_data.bid_price and market_data.ask_price:
                market_data.spread = round(market_data.ask_price - market_data.bid_price, 4)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error processing DataBento quote data for {symbol}: {str(e)}")
            return None
            
    def _process_quote_data(self, symbol: str, quote_response: Dict[str, Any]) -> Optional[MarketData]:
        """
        Process quote data from Polygon API into MarketData object
        
        Args:
            symbol (str): Stock symbol
            quote_response (Dict): Response from Polygon quote API
            
        Returns:
            Optional[MarketData]: Processed MarketData object or None if error
        """
        try:
            # Check if the response has results
            if 'results' not in quote_response:
                logger.warning(f"No results in quote_response for {symbol}")
                return None
            
            # For the prev aggs endpoint, the result is a list of bars
            results = quote_response['results']
            if not results:
                logger.warning(f"Empty results in quote_response for {symbol}")
                return None
                
            # Use the first (most recent) bar
            quote = results[0]
            
            # Create new MarketData object with data from the previous day bar
            # Use April 2023 date for timestamp instead of future date (2025)
            market_date = datetime.now() + timedelta(seconds=random.randint(0, 3600))
            
            market_data = MarketData(
                symbol=symbol,
                timestamp=market_date,
                # Use close as current price, low as bid, high as ask (simulation)
                bid_price=quote.get('l'),  # Low as bid
                bid_size=int(quote.get('v', 0) / 10),  # Fraction of volume as bid size
                ask_price=quote.get('h'),  # High as ask
                ask_size=int(quote.get('v', 0) / 10),  # Fraction of volume as ask size
                last_price=quote.get('c'),  # Close price
                volume=quote.get('v'),     # Volume
                vwap=quote.get('vw'),      # VWAP if available
                open_price=quote.get('o'), # Open price
                high_price=quote.get('h'), # High price
                low_price=quote.get('l'),  # Low price
                close_price=quote.get('c') # Close price
            )
            
            # Calculate spread if both bid and ask are available
            if market_data.bid_price and market_data.ask_price:
                market_data.spread = round(market_data.ask_price - market_data.bid_price, 4)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error processing quote data for {symbol}: {str(e)}")
            return None
    
    def _process_trade_data(self, symbol: str, trade_data: Dict[str, Any]) -> Optional[TradeData]:
        """
        Process trade data from Polygon API into TradeData object
        
        Args:
            symbol (str): Stock symbol
            trade_data (Dict): Trade data from Polygon API
            
        Returns:
            Optional[TradeData]: Processed TradeData object or None if error
        """
        try:
            # Use April 2023 date for timestamp instead of future date (2025)
            # Base date plus some random seconds to distribute the trades
            trade_time = datetime.now() + timedelta(seconds=random.randint(0, 3600))
            
            # If timestamp is in the data, use its time components but keep April 2023 date
            if 't' in trade_data:
                # Handle timestamp based on magnitude (ms vs ns)
                timestamp_value = trade_data['t']
                if timestamp_value > 1000000000000:  # Nanoseconds
                    original_time = datetime.fromtimestamp(timestamp_value / 1000000000)
                else:  # Milliseconds
                    original_time = datetime.fromtimestamp(timestamp_value / 1000)
                    
                # Keep April 2023 date but use original hour/minute/second
                trade_time = datetime(2023, 4, 11, 
                                     original_time.hour, 
                                     original_time.minute, 
                                     original_time.second)
            
            # Create TradeData object
            trade = TradeData(
                symbol=symbol,
                timestamp=trade_time,
                price=trade_data.get('p') or trade_data.get('c'),  # Price or close
                size=trade_data.get('s') or trade_data.get('v', 0),  # Size or volume
                exchange=trade_data.get('x') or 'POLY',  # Exchange or default
                trade_id=trade_data.get('i') or f"poly_{int(time.time())}_{symbol}",  # Trade ID or generate one
                tape=trade_data.get('z')  # Tape
            )
            
            # Determine if buyer was maker (default to None if unknown)
            if 'o' in trade_data:  # If order info available
                trade.is_buyer_maker = bool(trade_data.get('o') == 1)  # 1 = buy order at bid
            
            # Store the trade in the database
            self._save_trade_data(trade)
            
            return trade
            
        except Exception as e:
            logger.error(f"Error processing trade data for {symbol}: {str(e)}")
            return None
    
    def _save_market_data(self, market_data: MarketData) -> bool:
        """
        Save market data to the database
        
        Args:
            market_data (MarketData): MarketData object to save
            
        Returns:
            bool: Success status
        """
        if not market_data:
            return False
            
        try:
            db.session.add(market_data)
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving market data: {str(e)}")
            return False
    
    def _save_trade_data(self, trade_data: TradeData) -> bool:
        """
        Save trade data to the database
        
        Args:
            trade_data (TradeData): TradeData object to save
            
        Returns:
            bool: Success status
        """
        if not trade_data:
            return False
            
        try:
            # Check if this trade_id already exists to avoid duplicates
            if trade_data.trade_id:
                existing = TradeData.query.filter_by(trade_id=trade_data.trade_id).first()
                if existing:
                    return True  # Already stored this trade
            
            db.session.add(trade_data)
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving trade data: {str(e)}")
            return False
    
    def _get_latest_market_data(self, symbol: str) -> Optional[MarketData]:
        """
        Get the most recent market data for a symbol from the database
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Optional[MarketData]: Most recent MarketData object or None
        """
        try:
            return MarketData.query.filter_by(symbol=symbol).order_by(MarketData.timestamp.desc()).first()
        except Exception as e:
            logger.error(f"Error fetching latest market data for {symbol}: {str(e)}")
            return None
    
    def _get_recent_trades(self, symbol: str, limit: int = 100) -> List[TradeData]:
        """
        Get recent trades for a symbol from the database
        
        Args:
            symbol (str): Stock symbol
            limit (int): Maximum number of trades to return
            
        Returns:
            List[TradeData]: Recent trades
        """
        try:
            return TradeData.query.filter_by(symbol=symbol).order_by(TradeData.timestamp.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"Error fetching recent trades for {symbol}: {str(e)}")
            return []
    
    def _calculate_pressure(self, trades: List[TradeData]) -> Tuple[float, float]:
        """
        Calculate buying and selling pressure from recent trades
        
        Args:
            trades (List[TradeData]): List of recent trades
            
        Returns:
            Tuple[float, float]: (buying_pressure, selling_pressure)
        """
        if not trades:
            return 0.0, 0.0
            
        buy_volume = 0
        sell_volume = 0
        buy_count = 0
        sell_count = 0
        
        for trade in trades:
            if trade.is_buyer_maker is False:  # Trade hit the ask (buying)
                buy_volume += trade.size
                buy_count += 1
            elif trade.is_buyer_maker is True:  # Trade hit the bid (selling)
                sell_volume += trade.size
                sell_count += 1
        
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0.0, 0.0
            
        buying_pressure = round(buy_volume / total_volume * 100, 2) if total_volume > 0 else 0.0
        selling_pressure = round(sell_volume / total_volume * 100, 2) if total_volume > 0 else 0.0
        
        return buying_pressure, selling_pressure
    
    def _calculate_trade_speed(self, trades: List[TradeData]) -> float:
        """
        Calculate average number of trades per minute
        
        Args:
            trades (List[TradeData]): List of recent trades
            
        Returns:
            float: Average trades per minute
        """
        if not trades or len(trades) < 2:
            return 0.0
            
        # Get timestamps of newest and oldest trades
        newest = trades[0].timestamp
        oldest = trades[-1].timestamp
        
        # Calculate time span in minutes
        time_span = (newest - oldest).total_seconds() / 60.0
        
        if time_span <= 0:
            return 0.0
            
        return round(len(trades) / time_span, 2)
    
    def _determine_price_trend(self, trades: List[TradeData]) -> str:
        """
        Determine the price trend based on recent trades
        
        Args:
            trades (List[TradeData]): List of recent trades
            
        Returns:
            str: Price trend (up, down, sideways)
        """
        if not trades or len(trades) < 5:
            return "insufficient data"
            
        prices = [trade.price for trade in trades]
        
        # Calculate the trend using linear regression slope
        # Simple approach: compare latest vs earliest
        latest_avg = sum(prices[:5]) / 5.0
        earliest_avg = sum(prices[-5:]) / 5.0
        
        diff = latest_avg - earliest_avg
        
        if diff > 0.01:
            return "up"
        elif diff < -0.01:
            return "down"
        else:
            return "sideways"
    
    def _generate_recommendation(
        self, 
        market_data: MarketData, 
        buying_pressure: float, 
        selling_pressure: float, 
        trades_per_minute: float, 
        price_trend: str
    ) -> str:
        """
        Generate a trading recommendation based on analysis
        
        Args:
            market_data (MarketData): Latest market data
            buying_pressure (float): Buying pressure percentage
            selling_pressure (float): Selling pressure percentage
            trades_per_minute (float): Average trades per minute
            price_trend (str): Price trend direction
            
        Returns:
            str: Trading recommendation (Enter, Exit, Hold, Wait)
        """
        # Default to Wait when unsure
        if price_trend == "insufficient data":
            return "Wait"
            
        # Strong buy signal
        if buying_pressure > 65 and price_trend == "up" and trades_per_minute > 10:
            return "Enter"
            
        # Strong sell signal
        if selling_pressure > 65 and price_trend == "down" and trades_per_minute > 10:
            return "Exit"
            
        # Hold (already in position)
        if price_trend == "up" and buying_pressure > selling_pressure:
            return "Hold"
            
        # More conservative signals
        if buying_pressure > 55 and price_trend == "up":
            return "Enter"
            
        if selling_pressure > 55 and price_trend == "down":
            return "Exit"
            
        # Default to Wait
        return "Wait"

# Initialize a global instance
market_service = MarketDataService()