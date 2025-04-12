"""
Claude AI-powered market data analysis for trading application
Uses Anthropic Claude API to generate trading insights
"""

import os
import logging
import json
import anthropic
from typing import Dict, Any, List, Optional

# Set up logging
logger = logging.getLogger(__name__)

# Check for API key
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Create client if key is available
if ANTHROPIC_API_KEY:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
else:
    client = None
    logger.warning("ANTHROPIC_API_KEY not found. Claude analysis will not be available.")

def analyze_market_data(symbol: str, market_data: Dict[str, Any], 
                        time_and_sales: List[Dict[str, Any]] = None, 
                        order_book: Dict[str, Any] = None,
                        analysis_depth: str = 'standard') -> Dict[str, Any]:
    """
    Analyze market data using Claude AI to generate trading insights
    
    Args:
        symbol (str): Stock symbol
        market_data (Dict): Current market data including bid/ask prices
        time_and_sales (List): Recent trades data
        order_book (Dict): Level 2 order book data
        analysis_depth (str): Depth of analysis (basic, standard, detailed)
        
    Returns:
        Dict: Analysis result with Claude insights
    """
    logger.debug(f"Analyzing market data for {symbol} with Claude AI")
    
    # Check if Claude API key is available
    if not client:
        return {
            "success": False,
            "error": "Anthropic API key is required for Claude AI analysis",
            "symbol": symbol,
            "recommendation": "Wait",
            "confidence": 0.0,
            "insights": [f"Claude AI analysis unavailable - API key required"],
            "analysis": f"Please provide an Anthropic API key to get Claude AI-powered analysis for {symbol}"
        }
    
    # If market_data is None or empty, provide a default structure
    if not market_data:
        market_data = {"bid_price": 0, "ask_price": 0, "bid_size": 0, "ask_size": 0}
        
    # If time_and_sales is None, provide an empty list
    if time_and_sales is None:
        time_and_sales = []
        
    # If order_book is None, provide a default structure
    if order_book is None:
        order_book = {"bids": [], "asks": []}
    
    # Analysis depth settings
    depth_settings = {
        'basic': "Provide a brief analysis with just the essential details - focus on bid/ask, price action, and a simple recommendation.",
        'standard': "Provide a standard analysis with moderate detail - include order flow, spread implications, and a well-reasoned recommendation.",
        'detailed': "Provide an in-depth analysis with maximum detail - include pattern recognition, comprehensive order flow analysis, volume analysis, and a detailed trading strategy recommendation."
    }
    
    depth_guide = depth_settings.get(analysis_depth, depth_settings['standard'])
    
    # Format market data for the prompt
    market_data_text = f"""
    Symbol: {symbol}
    Current Price: ${market_data.get('last_price', 'N/A')}
    Bid: ${market_data.get('bid_price', 'N/A')} (Size: {market_data.get('bid_size', 'N/A')})
    Ask: ${market_data.get('ask_price', 'N/A')} (Size: {market_data.get('ask_size', 'N/A')})
    Spread: ${market_data.get('spread', 'N/A')}
    Volume: {market_data.get('volume', 'N/A')}
    VWAP: ${market_data.get('vwap', 'N/A')}
    Open: ${market_data.get('open_price', 'N/A')}
    High: ${market_data.get('high_price', 'N/A')}
    Low: ${market_data.get('low_price', 'N/A')}
    Close: ${market_data.get('close_price', 'N/A')}
    """
    
    # Format Time & Sales data if available
    time_sales_text = ""
    if time_and_sales and len(time_and_sales) > 0:
        time_sales_text = "Recent Trades (Time & Sales - newest first):\n"
        # Limit to 20 most recent trades to avoid prompt size issues
        for i, trade in enumerate(time_and_sales[:20]):
            time_sales_text += f"Trade {i+1}: Price=${trade.get('price', 'N/A')}, Size={trade.get('size', 'N/A')}, " \
                               f"Buyer Maker={trade.get('is_buyer_maker', 'Unknown')}\n"
    
    # Format Order Book data if available
    order_book_text = ""
    if order_book:
        order_book_text = "Current Order Book (Level 2 Data):\n"
        
        if 'asks' in order_book and order_book['asks']:
            order_book_text += "Asks (Sell Orders - lowest first):\n"
            for i, level in enumerate(order_book['asks'][:10]):  # Limit to top 10 levels
                order_book_text += f"  ${level.get('price', 'N/A')}: Size {level.get('size', 'N/A')}\n"
                
        if 'bids' in order_book and order_book['bids']:
            order_book_text += "Bids (Buy Orders - highest first):\n"
            for i, level in enumerate(order_book['bids'][:10]):  # Limit to top 10 levels
                order_book_text += f"  ${level.get('price', 'N/A')}: Size {level.get('size', 'N/A')}\n"
    
    # Prepare trading-specific instructions
    prompt = f"""
    You are an expert trading analyst specializing in market microstructure, order flow analysis, and price action trading.
    
    Analyze this real-time market data for {symbol} to provide concise trading insights.
    {depth_guide}
    
    Specifically analyze:
    
    1. Bid-Ask Spread: Note the current spread
    2. Order Stacking: Check for large orders at specific price levels
    3. Buying/Selling Pressure: Determine dominant direction  
    4. Price Action: Note key support/resistance levels
    5. Overall Market Sentiment: Bullish/Bearish/Neutral
    
    Based on your analysis, provide a clear trading recommendation (Enter, Exit, Hold, or Wait)
    with a VERY brief explanation (max 1-2 sentences). 
    
    Format your response as JSON with the following fields:
    - symbol: "{symbol}"
    - current_price: current price value
    - bid_price: bid price value
    - ask_price: ask price value
    - spread: spread value
    - analysis: KEEP THIS VERY CONCISE - max 3 short bullet points with key insights
    - recommendation: Enter/Exit/Hold/Wait
    - reason: 1-2 sentence explanation only
    - risk_level: Low/Medium/High
    
    IMPORTANT: Make your analysis extremely concise and easy to read quickly. Use short sentences and focus only on the most critical information a trader needs to know.
    
    Market Data:
    {market_data_text}
    {time_sales_text}
    {order_book_text}
    """
    
    try:
        # Make API call to Anthropic Claude
        # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            temperature=0.0,
            system="You are an expert trading analyst specializing in market microstructure, order flow analysis, and price action trading.",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        
        # Extract response content
        response_content = response.content[0].text
        
        # Try to parse response as JSON
        try:
            analysis_json = json.loads(response_content)
            logger.info(f"Successfully received trading analysis from Claude for {symbol}")
            
            # Add success flag to response
            analysis_json["success"] = True
            analysis_json["ai_analysis"] = True
            analysis_json["ai_provider"] = "Anthropic Claude"
            return analysis_json
        except json.JSONDecodeError:
            # If response is not valid JSON, extract key information manually
            logger.warning(f"Claude response is not valid JSON for {symbol}. Extracting information manually.")
            
            # Extract recommendation
            recommendation = "Wait"  # Default
            if "Enter" in response_content:
                recommendation = "Enter"
            elif "Exit" in response_content:
                recommendation = "Exit"
            elif "Hold" in response_content:
                recommendation = "Hold"
            
            # Create a structured response
            return {
                "success": True,
                "symbol": symbol,
                "current_price": market_data.get('last_price'),
                "bid_price": market_data.get('bid_price'),
                "ask_price": market_data.get('ask_price'),
                "spread": market_data.get('spread'),
                "analysis": response_content[:500],  # Limit to 500 chars
                "recommendation": recommendation,
                "reason": "See analysis for details",
                "risk_level": "Medium",
                "ai_analysis": True,
                "ai_provider": "Anthropic Claude (extracted from text)"
            }
            
    except Exception as e:
        logger.error(f"Error during Claude market data analysis: {str(e)}")
        # Return error response
        return {
            "success": False,
            "error": f"Claude analysis failed: {str(e)}",
            "symbol": symbol,
            "recommendation": "Wait",
            "analysis": "AI analysis unavailable. Using standard algorithmic analysis instead."
        }

def analyze_chart_patterns(symbol: str, price_history: List[Dict[str, Any]], 
                          timeframe: str = 'daily') -> Dict[str, Any]:
    """
    Analyze chart patterns using Claude AI
    
    Args:
        symbol (str): Stock symbol
        price_history (list): Historical price data (OHLCV)
        timeframe (str): Time interval (1m, 5m, 15m, 1h, daily, weekly)
        
    Returns:
        Dict: Pattern analysis with identified chart patterns
    """
    logger.debug(f"Analyzing chart patterns for {symbol} with Claude AI")
    
    # Check if Claude API key is available
    if not client:
        return {
            "success": False,
            "error": "Anthropic API key is required for Claude AI analysis",
            "symbol": symbol
        }
    
    # Format price history for the prompt
    price_data = ""
    if price_history and len(price_history) > 0:
        price_data = "Historical Price Data (most recent first):\n"
        # Limit to 30 most recent periods to keep prompt size reasonable
        for i, bar in enumerate(price_history[:30]):
            price_data += f"Period {i+1}: Open=${bar.get('open', 'N/A')}, High=${bar.get('high', 'N/A')}, " \
                         f"Low=${bar.get('low', 'N/A')}, Close=${bar.get('close', 'N/A')}, " \
                         f"Volume={bar.get('volume', 'N/A')}\n"
    
    # Prepare prompt for chart pattern analysis
    prompt = f"""
    You are an expert in technical analysis and chart pattern recognition.
    
    Analyze this {timeframe} price history for {symbol} and identify any significant chart patterns.
    
    Focus on identifying these patterns:
    1. Trend Analysis: Identify the primary trend (bullish, bearish, sideways)
    2. Support/Resistance Levels: Identify key price levels
    3. Chart Patterns: Identify any of these patterns if present:
       - Head and Shoulders / Inverse Head and Shoulders
       - Double Top / Double Bottom
       - Triple Top / Triple Bottom
       - Cup and Handle
       - Flags / Pennants
       - Wedges (Rising, Falling)
       - Triangles (Ascending, Descending, Symmetric)
       - Rectangles
    4. Candlestick Patterns: Identify significant candlestick patterns
    5. Indicator Signals: Based on price action, what would indicators likely show?
    
    Format your response as JSON with the following fields:
    - symbol: "{symbol}"
    - timeframe: "{timeframe}"
    - primary_trend: "bullish/bearish/sideways"
    - key_levels: [array of support/resistance price levels]
    - patterns_identified: [array of identified patterns]
    - pattern_confidence: high/medium/low
    - trading_outlook: bullish/bearish/neutral
    - recommendation: concise trading recommendation based on patterns
    
    Price History:
    {price_data}
    """
    
    try:
        # Make API call to Anthropic Claude
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            temperature=0.0,
            system="You are an expert technical analyst specializing in chart pattern recognition.",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        
        # Extract response content
        response_content = response.content[0].text
        
        # Try to parse response as JSON
        try:
            analysis_json = json.loads(response_content)
            logger.info(f"Successfully received chart pattern analysis from Claude for {symbol}")
            
            # Add success flag to response
            analysis_json["success"] = True
            return analysis_json
        except json.JSONDecodeError:
            # If response is not valid JSON, extract key information manually
            logger.warning(f"Claude response is not valid JSON for {symbol}. Returning text response.")
            
            return {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "analysis": response_content,
                "ai_provider": "Anthropic Claude"
            }
            
    except Exception as e:
        logger.error(f"Error during Claude chart pattern analysis: {str(e)}")
        return {
            "success": False,
            "error": f"Claude analysis failed: {str(e)}",
            "symbol": symbol
        }