import os
import logging
import json
from openai import OpenAI

# Set up logging
logger = logging.getLogger(__name__)

# Initialize the OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai = OpenAI(api_key=OPENAI_API_KEY)

def analyze_screenshot(base64_image, extracted_text, platform_type='generic', analysis_depth='standard'):
    """
    Analyze the screenshot using OpenAI's GPT-4o model.
    
    Args:
        base64_image (str): Base64 encoded screenshot
        extracted_text (str): Text extracted from the screenshot using OCR
        platform_type (str): Type of trading platform (generic, thinkorswim, tradingview, etc.)
        analysis_depth (str): Level of analysis detail (basic, standard, detailed)
        
    Returns:
        str: Analysis results from OpenAI
    """
    logger.debug("Analyzing screenshot with OpenAI")

    # Check if OpenAI API key is available
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key not available. Using fallback analysis.")
        return {
            "success": False,
            "error": "OpenAI API key required for AI analysis",
            "timestamp": datetime.now().isoformat(),
            "recommendation": "Wait",
            "confidence": 0.0,
            "analysis": "Please provide an OpenAI API key to access AI-powered analysis.",
            "insights": ["AI analysis unavailable - API key required"]
        }

    # Platform-specific instructions
    platform_instructions = {
        'thinkorswim': """
        For ThinkOrSwim platform:
        - Look for Time & Sales data in the right column (shows green for buys, red for sells)
        - Check for Level 2 data showing multiple bids/asks at different price levels
        - Examine depth of market which shows total size at various price points
        - ThinkOrSwim uses yellowish highlighting for key price levels
        - Look for tape speed which shows rate of trading volume
        """,
        'tradingview': """
        For TradingView platform:
        - Examine the order book/DOM if visible to identify buying/selling pressure
        - Look for chart pattern formations like triangles, flags, or head-and-shoulders
        - Check RSI, MACD, and other indicators visible in the bottom panels
        - Interpret volume bars at the bottom of the screen
        """,
        'mt4': """
        For MetaTrader 4:
        - Check the Market Watch window for bid/ask prices
        - Examine any visible indicators like Moving Averages, Bollinger Bands
        - Look for open positions visible in the Terminal window
        """,
        'mt5': """
        For MetaTrader 5:
        - Check the Depth of Market (DOM) window for order flow data
        - Look for Time & Sales data if visible
        - Examine any visible indicators and volume information
        """
    }
    
    # Analysis depth settings
    depth_settings = {
        'basic': "Provide a brief analysis with just the essential details - ticker symbol, current price, bid/ask, and a simple recommendation.",
        'standard': "Provide a standard analysis with moderate detail - include order flow, spread implications, and a well-reasoned recommendation.",
        'detailed': "Provide an in-depth analysis with maximum detail - include pattern recognition, comprehensive order flow analysis, volume analysis, and a detailed trading strategy recommendation."
    }
    
    # Select platform-specific instructions or use generic ones
    platform_guide = platform_instructions.get(platform_type, "")
    depth_guide = depth_settings.get(analysis_depth, depth_settings['standard'])
    
    # Prepare trading-specific instructions
    prompt = f"""
    Analyze this trading screen capture and extracted text to provide detailed market insights.
    Focus on Level 2 market data and Time & Sales information. {depth_guide}
    
    {platform_guide}
    
    Specifically analyze:
    
    1. Bid-Ask Spread: Calculate the current spread and its implications
    2. Order Stacking: Identify if there are stacked orders (multiple orders at the same price level)
    3. Buying/Selling Pressure: Analyze if asks are getting hit (buying pressure) or bids getting hit (selling pressure)
    4. Transaction Speed: Determine if trades are executing rapidly or slowly
    5. Volume Analysis: Comment on the relative size of transactions
    
    Based on your analysis, provide a clear trading recommendation (Enter, Exit, Hold, or Wait)
    with a brief explanation. Format your response with clear sections and include relevant numbers.
    Symbol: [TICKER]
    Bid: $XX.XX
    Ask: $XX.XX
    Spread: $0.XX
    """
    
    try:
        # Make API call to OpenAI
        response = openai.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert trading analyst specializing in Level 2 data and Time & Sales analysis."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prompt}\n\nExtracted text from screen: {extracted_text}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
        )
        
        # Extract and return the analysis text
        analysis_text = response.choices[0].message.content
        logger.info("Successfully received analysis from OpenAI")
        return analysis_text
        
    except Exception as e:
        logger.error(f"Error during OpenAI analysis: {str(e)}")
        # Return fallback analysis instead of raising an exception
        logger.info("Using fallback analysis due to OpenAI API error")
        return fallback_analysis(extracted_text, platform_type)
        
def analyze_market_data(symbol, market_data, time_and_sales=None, order_book=None, analysis_depth='standard'):
    """
    Analyze real-time market data using OpenAI's GPT-4o model
    
    Args:
        symbol (str): Stock symbol
        market_data (dict): Current market data including bid/ask prices, volumes, etc.
        time_and_sales (list, optional): Recent trade data
        order_book (dict, optional): Level 2 order book data
        analysis_depth (str): Level of analysis detail (basic, standard, detailed)
        
    Returns:
        dict: Analysis results including AI insights and trading recommendation
    """
    logger.debug(f"Analyzing market data for {symbol} with OpenAI")
    
    # Check if OpenAI API key is available
    if not OPENAI_API_KEY:
        from datetime import datetime
        return {
            "success": False,
            "error": "OpenAI API key is required for AI analysis",
            "symbol": symbol,
            "recommendation": "Wait",
            "confidence": 0.0,
            "insights": [f"AI analysis unavailable - API key required"],
            "analysis": f"Please provide an OpenAI API key to get AI-powered analysis for {symbol}",
            "timestamp": datetime.now().isoformat()
        }
    
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
    
    # Prepare trading-specific instructions - concise version
    prompt = f"""
    Analyze this real-time market data for {symbol} to provide concise trading insights.
    {depth_guide}
    
    Specifically analyze:
    
    1. Bid-Ask Spread: Note the current spread
    2. Order Stacking: Check for large orders at specific price levels
    3. Buying/Selling Pressure: Determine dominant direction  
    4. Price Action: Note key support/resistance levels
    5. Overall Market Sentiment: Bullish/Bearish/Neutral
    
    Based on your analysis, provide a clear trading recommendation (Enter, Exit, Hold, or Wait)
    with a VERY brief explanation (max 1-2 sentences). Format your response in JSON as follows:
    {{
        "symbol": "{symbol}",
        "current_price": "current price value",
        "bid_price": "bid price value",
        "ask_price": "ask price value",
        "spread": "spread value",
        "analysis": "KEEP THIS VERY CONCISE - max 3 short bullet points with key insights",
        "recommendation": "Enter/Exit/Hold/Wait",
        "reason": "1-2 sentence explanation only",
        "risk_level": "Low/Medium/High"
    }}
    
    IMPORTANT: Make your analysis extremely concise and easy to read quickly. Use short sentences and focus only on the most critical information a trader needs to know.
    """
    
    try:
        # Make API call to OpenAI with all the data
        response = openai.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert trading analyst specializing in market microstructure, order flow analysis, and price action trading."
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n\nMarket Data:\n{market_data_text}\n{time_sales_text}\n{order_book_text}"
                }
            ],
            response_format={"type": "json_object"},
        )
        
        # Extract and process the JSON response
        analysis_json = json.loads(response.choices[0].message.content)
        logger.info(f"Successfully received trading analysis from OpenAI for {symbol}")
        
        # Add success flag to response
        analysis_json["success"] = True
        return analysis_json
        
    except Exception as e:
        logger.error(f"Error during OpenAI market data analysis: {str(e)}")
        # Return error response
        return {
            "success": False,
            "error": f"OpenAI analysis failed: {str(e)}",
            "symbol": symbol,
            "recommendation": "Wait",
            "analysis": "AI analysis unavailable. Using standard algorithmic analysis instead."
        }

def fallback_analysis(extracted_text, platform_type='generic'):
    """
    Provide a basic analysis when OpenAI is unavailable
    
    Args:
        extracted_text (str): Text extracted from the screenshot
        platform_type (str): Type of trading platform for specialized parsing
        
    Returns:
        str: Basic analysis based on extracted text
    """
    import re
    
    # Default values
    analysis = {
        'symbol': 'Unknown',
        'bid': None,
        'ask': None,
        'spread': None,
        'recommendation': 'Wait'
    }
    
    # Platform-specific regex patterns
    platform_patterns = {
        'thinkorswim': {
            'symbol': r'(?:Symbol|Ticker)[:\s]+([A-Z]+)',
            'price': r'(?:Last|Price)[:\s]+\$?(\d+\.\d+)',
            'bid': r'(?:Bid|BID)[:\s]+\$?(\d+\.\d+)',
            'ask': r'(?:Ask|ASK|Offer)[:\s]+\$?(\d+\.\d+)',
            'volume': r'(?:VOL|Volume)[:\s]+(\d+(?:[,.]\d+)?[KMB]?)'
        },
        'generic': {
            'symbol': r'\b([A-Z]{1,5})\b',
            'price': r'\$?(\d+\.\d+)',
            'bid': r'(?:Bid|BID)[:\s]+\$?(\d+\.\d+)',
            'ask': r'(?:Ask|ASK|Offer)[:\s]+\$?(\d+\.\d+)'
        }
    }
    
    # Select pattern set based on platform type or default to generic
    patterns = platform_patterns.get(platform_type, platform_patterns['generic'])
    
    # Try to extract symbol using platform-specific patterns
    symbol_match = re.search(patterns['symbol'], extracted_text, re.IGNORECASE)
    if symbol_match:
        analysis['symbol'] = symbol_match.group(1)
    else:
        # Fallback to generic symbol pattern if platform-specific fails
        symbol_match = re.search(r'\b([A-Z]{1,5})\b', extracted_text)
        if symbol_match:
            analysis['symbol'] = symbol_match.group(1)
    
    # Extract bid price using platform-specific pattern
    bid_match = re.search(patterns.get('bid', r'Bid[:\s]+\$?(\d+\.\d+)'), extracted_text, re.IGNORECASE)
    if bid_match:
        try:
            analysis['bid'] = float(bid_match.group(1))
        except ValueError:
            pass
    
    # Extract ask price using platform-specific pattern
    ask_match = re.search(patterns.get('ask', r'Ask[:\s]+\$?(\d+\.\d+)'), extracted_text, re.IGNORECASE)
    if ask_match:
        try:
            analysis['ask'] = float(ask_match.group(1))
        except ValueError:
            pass
    
    # If we couldn't find specific bid/ask patterns, try the generic approach
    if analysis['bid'] is None or analysis['ask'] is None:
        price_matches = re.findall(r'\$?\s*(\d+\.\d+)\s*(?:USD)?', extracted_text)
        if len(price_matches) >= 2:
            try:
                # Take first two numbers as potential bid/ask
                bid = float(price_matches[0])
                ask = float(price_matches[1])
                # Make sure bid is lower than ask
                if bid > ask:
                    bid, ask = ask, bid
                # Only update if not already set by specific patterns
                if analysis['bid'] is None:
                    analysis['bid'] = bid
                if analysis['ask'] is None:
                    analysis['ask'] = ask
            except ValueError:
                pass
    
    # Calculate spread if we have both bid and ask
    if analysis['bid'] is not None and analysis['ask'] is not None:
        analysis['spread'] = round(analysis['ask'] - analysis['bid'], 2)
    
    # Build response text
    response = f"**Basic Trading Screen Analysis**\n\n"
    response += f"Symbol: {analysis['symbol']}\n"
    
    if analysis['bid']:
        response += f"Bid: ${analysis['bid']}\n"
    else:
        response += "Bid: Not detected\n"
        
    if analysis['ask']:
        response += f"Ask: ${analysis['ask']}\n"
    else:
        response += "Ask: Not detected\n"
        
    if analysis['spread']:
        response += f"Spread: ${analysis['spread']}\n"
    else:
        response += "Spread: Not calculated\n"
    
    response += "\n**Note:** This is basic analysis from OCR text only. For full AI analysis, please provide a valid OpenAI API key."
    
    return response