"""
WebSocket event handlers and callbacks for processing real-time market data
"""
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from trading_intelligence import trading_intelligence

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory storage for latest market data
latest_data = {}

def store_market_data(symbol: str, data_type: str, data: Dict[str, Any]):
    """
    Store the latest market data in memory
    
    Args:
        symbol (str): The trading symbol
        data_type (str): The type of data (l2, trades, quotes)
        data (Dict): The market data
    """
    if symbol not in latest_data:
        latest_data[symbol] = {}
    
    # Create a deep copy to avoid reference issues
    import copy
    data_copy = copy.deepcopy(data)
    
    latest_data[symbol][data_type] = {
        "data": data_copy,
        "timestamp": datetime.now().isoformat()
    }

def get_latest_data(symbol: str, data_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the latest market data from memory
    
    Args:
        symbol (str): The trading symbol
        data_type (str, optional): The type of data to get. If None, returns all data.
        
    Returns:
        Dict: The latest market data
    """
    if symbol not in latest_data:
        return {}
    
    if data_type:
        return latest_data[symbol].get(data_type, {})
    
    return latest_data[symbol]

async def handle_level2_data(data: Dict[str, Any]):
    """
    Handle Level 2 market depth data and generate insights
    
    Args:
        data (Dict): The Level 2 data from the WebSocket
    """
    try:
        symbol = data.get('symbol', 'Unknown')
        
        # Extract bid and ask levels
        bids = []
        asks = []
        
        # Process MBP-10 data format
        if data.get('schema') == 'mbp_10':
            for level in range(10):
                bid_price_key = f'bid_px_{level:02d}'
                bid_size_key = f'bid_sz_{level:02d}'
                ask_price_key = f'ask_px_{level:02d}'
                ask_size_key = f'ask_sz_{level:02d}'
                
                if bid_price_key in data and data[bid_price_key] > 0:
                    bids.append({
                        'price': data[bid_price_key],
                        'size': data[bid_size_key]
                    })
                
                if ask_price_key in data and data[ask_price_key] > 0:
                    asks.append({
                        'price': data[ask_price_key],
                        'size': data[ask_size_key]
                    })
        
        # Store the processed data
        l2_data = {
            "bids": bids,
            "asks": asks,
            "symbol": symbol,
            "source": "databento_websocket",
            "timestamp": datetime.now().isoformat()
        }
        store_market_data(symbol, "l2", l2_data)
        
        # Get recent trades for enhanced analysis
        trades_data = get_latest_data(symbol, "trades")
        recent_trades = trades_data.get("data", {}).get("trades", [])[:10] if trades_data else []
        
        # Generate insights if we have enough data
        if bids and asks:
            # Enhanced pro-level analysis
            ask_hits = [p for p in recent_trades if p.get("side", "") == "buy"]
            bid_hits = [p for p in recent_trades if p.get("side", "") == "sell"]
            
            # Track aggressive trade activity
            if ask_hits and bid_hits:
                ask_volume = sum(p.get("size", 0) for p in ask_hits)
                bid_volume = sum(p.get("size", 0) for p in bid_hits)
                if len(ask_hits) > len(bid_hits) and ask_volume > 2000:
                    l2_data["aggressive_buying"] = True
                    logger.info(f"🚀 Large size printing into the ask — aggressive buyers stepping in. ({ask_volume} shares)")
                elif len(bid_hits) > len(ask_hits) and bid_volume > 2000:
                    l2_data["aggressive_selling"] = True
                    logger.info(f"🔻 Heavy selling into bid — watch for a flush. ({bid_volume} shares)")
            
            # Check for spread tightening
            spread = round(asks[0]["price"] - bids[0]["price"], 2)
            if spread <= 0.01:
                l2_data["tight_spread"] = True
                logger.info(f"🟢 Spread tightening (${spread}) — breakout conditions building.")
            
            # Check for bid stacking
            bid_sizes = [b["size"] for b in bids[:5]]
            avg_bid = sum(bid_sizes) / len(bid_sizes) if bid_sizes else 0
            if any(size > 4 * avg_bid for size in bid_sizes):
                l2_data["stacked_bids"] = True
                logger.info(f"📈 Strong bid stacking — buyers loading below.")
            
            # More comprehensive AI-powered analysis
            insights = trading_intelligence.analyze_order_book(symbol, bids, asks, recent_trades)
            if insights:
                logger.info(f"L2 Insights for {symbol}: {insights['messages']}")
                logger.info(f"Recommendation: {insights['recommendation']} (Confidence: {insights['confidence']:.2f})")
                
                # Store insights
                l2_data["insights"] = insights
        
    except Exception as e:
        logger.error(f"Error processing Level 2 data: {str(e)}")

async def handle_trade_data(data: Dict[str, Any]):
    """
    Handle trade data and generate insights
    
    Args:
        data (Dict): The trade data from the WebSocket
    """
    try:
        symbol = data.get('symbol', 'Unknown')
        
        # Create a standardized trade record
        price = data.get('price', 0)
        size = data.get('size', 0)
        side = data.get('side', '')
        is_buyer_maker = side == 'B'
        
        trade = {
            "price": price,
            "size": size,
            "is_buyer_maker": is_buyer_maker,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "source": "databento_websocket"
        }
        
        # Get existing trades or initialize
        trades_data = get_latest_data(symbol, "trades")
        trades_list = trades_data.get("data", {}).get("trades", []) if trades_data else []
        
        # Add new trade and keep only recent ones
        trades_list.insert(0, trade)  # Add to beginning (newest first)
        trades_list = trades_list[:50]  # Keep only the most recent 50 trades
        
        # Store updated trades
        store_market_data(symbol, "trades", {
            "trades": trades_list,
            "symbol": symbol,
            "count": len(trades_list),
            "source": "databento_websocket",
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate insights if we have enough trades
        if len(trades_list) >= 5:
            insights = trading_intelligence.analyze_trades(symbol, trades_list[:10])  # Analyze the 10 most recent trades
            if insights:
                logger.info(f"Trades Insights for {symbol}: {insights['messages']}")
                logger.info(f"Recommendation: {insights['recommendation']} (Confidence: {insights['confidence']:.2f})")
                
    except Exception as e:
        logger.error(f"Error processing trade data: {str(e)}")

async def handle_quote_data(data: Dict[str, Any]):
    """
    Handle quote data (BBO)
    
    Args:
        data (Dict): The quote data from the WebSocket
    """
    try:
        symbol = data.get('symbol', 'Unknown')
        
        # Extract NBBO data from BBO_1S format
        bid_price = data.get('bid_px_00', 0)
        bid_size = data.get('bid_sz_00', 0)
        ask_price = data.get('ask_px_00', 0)
        ask_size = data.get('ask_sz_00', 0)
        
        # Calculate spread
        spread = round(ask_price - bid_price, 4) if ask_price > 0 and bid_price > 0 else 0
        
        quote = {
            "bid_price": bid_price,
            "bid_size": bid_size,
            "ask_price": ask_price,
            "ask_size": ask_size,
            "spread": spread,
            "timestamp": datetime.now().isoformat()
        }
        
        # Get existing quotes or initialize
        quotes_data = get_latest_data(symbol, "quotes")
        quotes_list = quotes_data.get("data", {}).get("quotes", []) if quotes_data else []
        
        # Add new quote and keep only recent ones
        quotes_list.insert(0, quote)  # Add to beginning (newest first)
        quotes_list = quotes_list[:50]  # Keep only the most recent 50 quotes
        
        # Store updated quotes
        store_market_data(symbol, "quotes", {
            "quotes": quotes_list,
            "market_data": quote,  # Current quote as market data
            "symbol": symbol,
            "count": len(quotes_list),
            "source": "databento_websocket",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing quote data: {str(e)}")

def get_combined_insights(symbol: str) -> Dict[str, Any]:
    """
    Get combined insights for a symbol from both order book and trades analysis
    
    Args:
        symbol (str): The trading symbol
        
    Returns:
        Dict: Combined insights
    """
    # Get latest insights
    insights = trading_intelligence.get_last_insights(symbol)
    
    if not insights:
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "messages": ["No significant patterns detected yet."],
            "recommendation": "Wait",
            "confidence": 0.0
        }
    
    return insights

def get_tape_reading_data(symbol: str) -> Dict[str, Any]:
    """
    Get comprehensive tape reading data for a symbol
    Combines Level 2, trades, quotes and insights
    
    Args:
        symbol (str): The trading symbol
        
    Returns:
        Dict: Comprehensive tape reading data
    """
    symbol = symbol.upper()
    
    # Get data from storage
    l2_data = get_latest_data(symbol, "l2").get("data", {})
    trades_data = get_latest_data(symbol, "trades").get("data", {})
    quotes_data = get_latest_data(symbol, "quotes").get("data", {})
    
    # Get combined insights
    insights = get_combined_insights(symbol)
    
    return {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "market_depth": l2_data,
        "trades": trades_data.get("trades", [])[:10],  # Top 10 trades
        "quotes": quotes_data.get("quotes", [])[:5],   # Top 5 quotes
        "market_data": quotes_data.get("market_data", {}),
        "insights": insights,
        "source": "databento_websocket"
    }