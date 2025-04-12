import logging
import json
import os
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
from market_data_service import market_service
from databento_client import databento_client
from databento_advanced import databento_advanced
from ib_client import ib_client
from realtime_data_simulator import realtime_simulator
import anthropic_analyzer
from trading_intelligence import trading_intelligence
from market_data_factory import REALTIME_ENABLED, create_market_data_feed, get_data_source_info

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a Blueprint for market data routes
market_bp = Blueprint('market', __name__, url_prefix='/market')

# Global WebSocket client instance
from websocket_handlers import handle_level2_data, handle_trade_data, handle_quote_data, get_tape_reading_data, get_combined_insights
# Use DatabentoWebSocket as the implementation for backwards compatibility
from databento_websocket import DatabentoWebSocket
websocket_client = None
websocket_task = None

@market_bp.route('/data/<symbol>', methods=['GET'])
def get_market_data(symbol):
    """
    Get real-time market data for a specific symbol
    
    Args:
        symbol (str): The stock symbol to get data for
        
    Returns:
        JSON: Market data response
    """
    symbol = symbol.upper().strip()
    
    # Add to active symbols if not already monitored
    if symbol not in market_service.get_active_symbols():
        market_service.add_symbol(symbol)
    
    # Fetch the latest market data
    response = market_service.get_market_data(symbol)
    return jsonify(response)

@market_bp.route('/monitor', methods=['POST'])
def add_symbol():
    """
    Add a symbol to active monitoring
    
    Returns:
        JSON: Response with success status and initial data
    """
    data = request.get_json()
    
    if not data or 'symbol' not in data:
        return jsonify({"success": False, "error": "Symbol is required"}), 400
    
    symbol = data['symbol'].upper().strip()
    
    # Add symbol to monitoring
    response = market_service.add_symbol(symbol)
    return jsonify(response)

@market_bp.route('/monitor/<symbol>', methods=['DELETE'])
def remove_symbol(symbol):
    """
    Remove a symbol from active monitoring
    
    Args:
        symbol (str): The stock symbol to stop monitoring
        
    Returns:
        JSON: Response with success status
    """
    symbol = symbol.upper().strip()
    
    # Remove symbol from monitoring
    response = market_service.remove_symbol(symbol)
    return jsonify(response)

@market_bp.route('/monitor', methods=['GET'])
def get_active_symbols():
    """
    Get the list of actively monitored symbols
    
    Returns:
        JSON: List of active symbols
    """
    symbols = market_service.get_active_symbols()
    return jsonify({"success": True, "symbols": symbols, "count": len(symbols)})

@market_bp.route('/time-and-sales/<symbol>', methods=['GET'])
def get_time_and_sales(symbol):
    """
    Get Time & Sales data (recent trades) for a symbol
    
    Args:
        symbol (str): The stock symbol
        
    Returns:
        JSON: Time & Sales data
    """
    symbol = symbol.upper().strip()
    
    # Get limit parameter (default: 50)
    try:
        limit = int(request.args.get('limit', 50))
        limit = min(max(1, limit), 1000)  # Ensure limit is between 1 and 1000
    except ValueError:
        limit = 50
    
    # Fetch Time & Sales data
    response = market_service.get_time_and_sales(symbol, limit)
    return jsonify(response)

@market_bp.route('/level2/<symbol>', methods=['GET'])
def get_level2_data(symbol):
    """
    Get Level 2 order book data for a symbol
    
    Args:
        symbol (str): The stock symbol
        
    Returns:
        JSON: Level 2 data
    """
    symbol = symbol.upper().strip()
    
    # Fetch Level 2 data
    response = market_service.get_order_book(symbol)
    return jsonify(response)

@market_bp.route('/bars/<symbol>/<interval>', methods=['GET'])
def get_historical_bars(symbol, interval):
    """
    Get historical bar data for a symbol
    
    Args:
        symbol (str): The stock symbol
        interval (str): Time interval ('minute', 'hour', 'day')
        
    Returns:
        JSON: Historical bar data
    """
    symbol = symbol.upper().strip()
    
    # Validate interval
    if interval not in ['minute', 'hour', 'day']:
        interval = 'day'  # Default to daily bars
    
    # Get limit parameter (default: 50)
    try:
        limit = int(request.args.get('limit', 50))
        limit = min(max(1, limit), 500)  # Ensure limit is between 1 and 500
    except ValueError:
        limit = 50
    
    # Fetch historical bars from DataBento
    response = databento_client.get_historical_bars(symbol, interval, limit)
    return jsonify(response)

@market_bp.route('/update-all', methods=['POST'])
def update_all():
    """
    Update market data for all actively monitored symbols
    
    Returns:
        JSON: Results of all updates
    """
    # Update all symbols
    response = market_service.update_all_symbols()
    return jsonify(response)

@market_bp.route('/analysis/<symbol>', methods=['GET'])
def get_analysis(symbol):
    """
    Get trading analysis for a symbol based on recent market data
    
    Args:
        symbol (str): The stock symbol
        
    Returns:
        JSON: Analysis results
    """
    symbol = symbol.upper().strip()
    
    # Get query parameters
    use_ai = request.args.get('use_ai', 'true').lower() == 'true'
    analysis_depth = request.args.get('depth', 'standard')
    
    # Validate analysis depth
    if analysis_depth not in ['basic', 'standard', 'detailed']:
        analysis_depth = 'standard'
    
    # Ensure we have recent data
    market_service.get_market_data(symbol)
    
    # Generate analysis with AI if requested
    response = market_service.get_analysis(
        symbol=symbol,
        use_ai=use_ai,
        analysis_depth=analysis_depth
    )
    
    return jsonify(response)

@market_bp.route('/ai-analysis/<symbol>', methods=['GET'])
def get_ai_analysis(symbol):
    """
    Get AI-powered trading analysis for a symbol using OpenAI
    
    Args:
        symbol (str): The stock symbol
        
    Returns:
        JSON: AI analysis results
    """
    symbol = symbol.upper().strip()
    
    # Get analysis depth parameter
    analysis_depth = request.args.get('depth', 'detailed')
    
    # Validate analysis depth
    if analysis_depth not in ['basic', 'standard', 'detailed']:
        analysis_depth = 'detailed'
    
    # Ensure we have recent data
    market_service.get_market_data(symbol)
    
    # Generate analysis with AI
    response = market_service.get_analysis(
        symbol=symbol,
        use_ai=True,
        analysis_depth=analysis_depth
    )
    
    # If AI analysis failed, provide a clear message
    if not response.get('success', False) or not response.get('ai_analysis', False):
        if 'error' not in response:
            response['error'] = "OpenAI analysis failed. Please check if you have a valid OpenAI API key."
    
    return jsonify(response)

@market_bp.route('/databento/<symbol>', methods=['GET'])
def test_databento(symbol):
    """
    Direct test of the DataBento client for a specific symbol
    
    Args:
        symbol (str): The stock symbol
        
    Returns:
        JSON: Combined DataBento API responses
    """
    symbol = symbol.upper().strip()
    
    # Get the test_type parameter
    test_type = request.args.get('type', 'all')
    
    results = {}
    
    if test_type in ['all', 'ticker']:
        # Test ticker details
        ticker_response = databento_client.get_ticker_details(symbol)
        results['ticker_details'] = ticker_response
    
    if test_type in ['all', 'quotes']:
        # Test NBBO quotes
        quotes_response = databento_client.get_nbbo_quotes(symbol)
        results['nbbo_quotes'] = quotes_response
    
    if test_type in ['all', 'level2']:
        # Test Level 2 order book
        level2_response = databento_client.get_level2_data(symbol)
        results['level2_data'] = level2_response
    
    if test_type in ['all', 'trades']:
        # Test Time & Sales
        trades_response = databento_client.get_time_and_sales(symbol)
        results['time_and_sales'] = trades_response
    
    if test_type in ['all', 'bars']:
        # Test historical bars
        bars_response = databento_client.get_historical_bars(symbol, 'day', 5)
        results['historical_bars'] = bars_response
    
    return jsonify({
        "success": True,
        "symbol": symbol,
        "data": results
    })
    
# Interactive Brokers API Routes

@market_bp.route('/ib/connect', methods=['POST'])
def ib_connect():
    """
    Connect to the Interactive Brokers TWS or Gateway
    
    Returns:
        JSON: Connection status
    """
    data = request.get_json() or {}
    
    # Get connection parameters with defaults
    host = data.get('host', '127.0.0.1')
    port = data.get('port', 7497)  # Default for TWS
    client_id = data.get('client_id', 1)
    read_only = data.get('read_only', True)
    
    # Connect to IB
    success = ib_client.connect(host, port, client_id, read_only)
    
    if success:
        return jsonify({
            "success": True,
            "message": f"Connected to Interactive Brokers on {host}:{port}",
            "connected": True
        })
    else:
        return jsonify({
            "success": False,
            "error": "Failed to connect to Interactive Brokers",
            "connected": False
        })

@market_bp.route('/ib/disconnect', methods=['POST'])
def ib_disconnect():
    """
    Disconnect from Interactive Brokers
    
    Returns:
        JSON: Disconnection status
    """
    success = ib_client.disconnect()
    
    return jsonify({
        "success": success,
        "message": "Disconnected from Interactive Brokers",
        "connected": False
    })

@market_bp.route('/ib/status', methods=['GET'])
def ib_status():
    """
    Check the connection status to Interactive Brokers
    
    Returns:
        JSON: Connection status
    """
    connected = ib_client.is_connected()
    
    return jsonify({
        "success": True,
        "connected": connected,
        "message": "Connected to Interactive Brokers" if connected else "Not connected to Interactive Brokers"
    })

@market_bp.route('/ib/market-data/<symbol>', methods=['GET'])
def ib_market_data(symbol):
    """
    Get market data from Interactive Brokers for a symbol
    
    Args:
        symbol (str): The stock symbol
        
    Returns:
        JSON: Market data response
    """
    symbol = symbol.upper().strip()
    
    # Get security type parameter (default: STK)
    security_type = request.args.get('type', 'STK').upper()
    exchange = request.args.get('exchange', 'SMART')
    currency = request.args.get('currency', 'USD')
    
    # Get additional parameters for specific security types
    additional_params = {}
    if security_type == 'OPT':
        additional_params['expiry'] = request.args.get('expiry', '')
        additional_params['strike'] = float(request.args.get('strike', 0.0))
        additional_params['right'] = request.args.get('right', 'C')
    elif security_type == 'FUT':
        additional_params['expiry'] = request.args.get('expiry', '')
    
    # Get market data from IB
    response = ib_client.get_market_data(
        symbol=symbol,
        security_type=security_type,
        exchange=exchange,
        currency=currency,
        **additional_params
    )
    
    return jsonify(response)

@market_bp.route('/ib/order-book/<symbol>', methods=['GET'])
def ib_order_book(symbol):
    """
    Get order book (market depth) from Interactive Brokers for a symbol
    
    Args:
        symbol (str): The stock symbol
        
    Returns:
        JSON: Order book data response
    """
    symbol = symbol.upper().strip()
    
    # Get security type parameter (default: STK)
    security_type = request.args.get('type', 'STK').upper()
    exchange = request.args.get('exchange', 'SMART')
    currency = request.args.get('currency', 'USD')
    
    # Get depth parameter
    try:
        depth = int(request.args.get('depth', 10))
        depth = min(max(1, depth), 20)  # Limit depth between 1 and 20
    except ValueError:
        depth = 10
    
    # Get additional parameters for specific security types
    additional_params = {}
    if security_type == 'OPT':
        additional_params['expiry'] = request.args.get('expiry', '')
        additional_params['strike'] = float(request.args.get('strike', 0.0))
        additional_params['right'] = request.args.get('right', 'C')
    elif security_type == 'FUT':
        additional_params['expiry'] = request.args.get('expiry', '')
    
    # Get order book from IB
    response = ib_client.get_order_book(
        symbol=symbol,
        security_type=security_type,
        exchange=exchange,
        currency=currency,
        depth=depth,
        **additional_params
    )
    
    return jsonify(response)

@market_bp.route('/ib/historical/<symbol>', methods=['GET'])
def ib_historical_data(symbol):
    """
    Get historical data from Interactive Brokers for a symbol
    
    Args:
        symbol (str): The stock symbol
        
    Returns:
        JSON: Historical data response
    """
    symbol = symbol.upper().strip()
    
    # Get security type parameter (default: STK)
    security_type = request.args.get('type', 'STK').upper()
    exchange = request.args.get('exchange', 'SMART')
    currency = request.args.get('currency', 'USD')
    
    # Get duration and bar size parameters
    duration = request.args.get('duration', '1 D')
    bar_size = request.args.get('bar_size', '1 min')
    
    # Validate bar size
    valid_bar_sizes = ['1 secs', '5 secs', '10 secs', '15 secs', '30 secs', 
                       '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins', '20 mins', '30 mins',
                       '1 hour', '2 hours', '3 hours', '4 hours', '8 hours',
                       '1 day', '1 week', '1 month']
    if bar_size not in valid_bar_sizes:
        bar_size = '1 min'  # Default to 1-minute bars
    
    # Get additional parameters for specific security types
    additional_params = {}
    if security_type == 'OPT':
        additional_params['expiry'] = request.args.get('expiry', '')
        additional_params['strike'] = float(request.args.get('strike', 0.0))
        additional_params['right'] = request.args.get('right', 'C')
    elif security_type == 'FUT':
        additional_params['expiry'] = request.args.get('expiry', '')
    
    # Get historical data from IB
    response = ib_client.get_historical_data(
        symbol=symbol,
        security_type=security_type,
        exchange=exchange,
        currency=currency,
        duration=duration,
        bar_size=bar_size,
        **additional_params
    )
    
    return jsonify(response)

@market_bp.route('/ib/cancel-market-data', methods=['POST'])
def ib_cancel_market_data():
    """
    Cancel market data subscription for a symbol or all symbols
    
    Returns:
        JSON: Cancellation status
    """
    data = request.get_json() or {}
    
    # Get symbol parameter (None will cancel all subscriptions)
    symbol = data.get('symbol')
    
    # Cancel market data subscription
    success = ib_client.cancel_market_data(symbol)
    
    return jsonify({
        "success": success,
        "message": f"Cancelled market data subscription for {symbol}" if symbol else "Cancelled all market data subscriptions"
    })
    
# Anthropic Claude AI Analysis Routes

@market_bp.route('/claude-analysis/<symbol>', methods=['GET'])
def get_claude_analysis(symbol):
    """
    Get AI-powered trading analysis for a symbol using both Anthropic Claude and pattern recognition
    
    Args:
        symbol (str): The stock symbol
        
    Returns:
        JSON: Combined analysis results including AI insights and pattern recognition
    """
    symbol = symbol.upper().strip()
    
    # Get analysis parameters
    analysis_depth = request.args.get('depth', 'detailed')
    analysis_type = request.args.get('type', 'combined')  # ai, pattern, combined
    
    # Validate analysis depth
    if analysis_depth not in ['basic', 'standard', 'detailed']:
        analysis_depth = 'detailed'
    
    # Ensure we have recent data
    market_data_response = market_service.get_market_data(symbol)
    
    if not market_data_response.get('success', False):
        return jsonify({
            "success": False,
            "error": "Failed to get market data",
            "symbol": symbol
        })
    
    market_data = market_data_response.get('market_data', {})
    
    # Get time and sales data
    time_sales_response = market_service.get_time_and_sales(symbol, 30)
    time_and_sales = time_sales_response.get('trades', []) if time_sales_response.get('success', False) else []
    
    # Get order book data
    order_book_response = market_service.get_order_book(symbol)
    order_book = None
    if order_book_response.get('success', False):
        order_book = {
            'bids': order_book_response.get('bids', []),
            'asks': order_book_response.get('asks', [])
        }
    
    # Build the response with all available analysis methods
    analysis_response = {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'market_data': market_data,
        'analysis_depth': analysis_depth,
        'analysis_type': analysis_type
    }
    
    # 1. Generate pattern-based trading intelligence analysis
    if analysis_type in ['pattern', 'combined']:
        try:
            pattern_analysis = trading_intelligence.get_advanced_insights(
                symbol=symbol,
                market_data=market_data,
                time_and_sales=time_and_sales,
                order_book=order_book
            )
            analysis_response['pattern_analysis'] = pattern_analysis
        except Exception as e:
            analysis_response['pattern_analysis_error'] = str(e)
    
    # 2. Generate AI-based analysis with Claude if available
    if analysis_type in ['ai', 'combined']:
        try:
            ai_analysis = anthropic_analyzer.analyze_market_data(
                symbol=symbol,
                market_data=market_data,
                time_and_sales=time_and_sales,
                order_book=order_book,
                analysis_depth=analysis_depth
            )
            analysis_response['ai_analysis'] = ai_analysis
        except Exception as e:
            analysis_response['ai_analysis_error'] = str(e)
    
    # 3. Add a summary section for quick reference
    summary = []
    
    # Add pattern-based insights to summary
    if 'pattern_analysis' in analysis_response:
        pattern_insights = analysis_response['pattern_analysis']
        if 'combined_messages' in pattern_insights:
            summary.extend(pattern_insights['combined_messages'][:3])  # Top 3 insights
        
        if 'final_recommendation' in pattern_insights:
            recommendation = pattern_insights['final_recommendation']
            summary.append(f"Pattern Analysis Recommendation: {recommendation['action']} (Confidence: {recommendation['confidence']:.2f})")
    
    # Add AI-based insights to summary
    if 'ai_analysis' in analysis_response:
        ai_insights = analysis_response['ai_analysis']
        if 'insights' in ai_insights:
            summary.extend(ai_insights.get('insights', [])[:3])  # Top 3 insights
        
        if 'recommendation' in ai_insights:
            summary.append(f"AI Analysis Recommendation: {ai_insights['recommendation']} (Confidence: {ai_insights.get('confidence', 0):.2f})")
    
    analysis_response['summary'] = summary
    
    return jsonify(analysis_response)

@market_bp.route('/chart-patterns/<symbol>', methods=['GET'])
def get_chart_patterns(symbol):
    """
    Get chart pattern analysis for a symbol using Anthropic Claude
    
    Args:
        symbol (str): The stock symbol
        
    Returns:
        JSON: Chart pattern analysis results
    """
    symbol = symbol.upper().strip()
    
    # Get timeframe parameter
    timeframe = request.args.get('timeframe', 'daily')
    
    # Validate timeframe
    valid_timeframes = ['1m', '5m', '15m', '1h', 'daily', 'weekly']
    if timeframe not in valid_timeframes:
        timeframe = 'daily'
    
    # Map timeframe to interval for historical data
    interval_map = {
        '1m': 'minute',
        '5m': 'minute',
        '15m': 'minute',
        '1h': 'hour',
        'daily': 'day',
        'weekly': 'day'
    }
    
    # Get historical data
    limit = 30  # Default to 30 bars
    if timeframe == '1m':
        limit = 60  # More data points for shorter timeframes
    elif timeframe in ['5m', '15m']:
        limit = 50
        
    interval = interval_map.get(timeframe, 'day')
    
    # Fetch historical data from DataBento or simulator
    bars_response = databento_client.get_historical_bars(symbol, interval, limit)
    
    if not bars_response.get('success', False):
        # Try simulator as fallback
        try:
            bars_response = realtime_simulator.get_historical_bars(symbol, interval, limit)
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return jsonify({
                "success": False,
                "error": "Failed to get historical data",
                "symbol": symbol
            })
    
    bars = bars_response.get('bars', [])
    
    if not bars:
        return jsonify({
            "success": False,
            "error": "No historical data available",
            "symbol": symbol
        })
    
    # Generate chart pattern analysis with Claude
    analysis = anthropic_analyzer.analyze_chart_patterns(
        symbol=symbol,
        price_history=bars,
        timeframe=timeframe
    )
    
    return jsonify(analysis)
    
# Advanced Databento Routes

@market_bp.route('/databento/advanced/datasets', methods=['GET'])
def get_available_datasets():
    """
    Get all available datasets from Databento
    
    Returns:
        JSON: Available datasets
    """
    response = databento_advanced.list_available_datasets()
    return jsonify(response)

@market_bp.route('/databento/advanced/symbols/<dataset>', methods=['GET'])
def list_symbols(dataset):
    """
    List available symbols for a dataset
    
    Args:
        dataset (str): Dataset name (e.g., 'XNAS_ITCH')
        
    Returns:
        JSON: Available symbols
    """
    # Get limit parameter (default: 100)
    try:
        limit = int(request.args.get('limit', 100))
        limit = min(max(1, limit), 1000)  # Ensure limit is between 1 and 1000
    except ValueError:
        limit = 100
    
    # Get symbol type parameter
    stype = request.args.get('stype')
    
    # List symbols
    response = databento_advanced.list_symbols(dataset, stype, limit)
    return jsonify(response)

@market_bp.route('/databento/advanced/market-depth/<symbol>', methods=['GET'])
def get_advanced_market_depth(symbol):
    """
    Get market depth data (order book) for a symbol using advanced Databento client
    
    Args:
        symbol (str): The stock symbol
        
    Returns:
        JSON: Market depth data
    """
    symbol = symbol.upper().strip()
    
    # Get dataset parameter
    dataset = request.args.get('dataset', 'XNAS_ITCH')
    
    # Get date parameter
    date = request.args.get('date')
    
    # Get depth parameter
    try:
        depth = int(request.args.get('depth', 10))
        depth = min(max(1, depth), 20)  # Ensure depth is between 1 and 20
    except ValueError:
        depth = 10
    
    # Get market depth data
    response = databento_advanced.get_market_depth(symbol, dataset, date, depth)
    return jsonify(response)

@market_bp.route('/databento/advanced/trades/<symbol>', methods=['GET'])
def get_advanced_trades(symbol):
    """
    Get trades (time & sales) data for a symbol using advanced Databento client
    
    Args:
        symbol (str): The stock symbol
        
    Returns:
        JSON: Trades data
    """
    symbol = symbol.upper().strip()
    
    # Get dataset parameter
    dataset = request.args.get('dataset', 'XNAS_ITCH')
    
    # Get date parameter
    date = request.args.get('date')
    
    # Get limit parameter
    try:
        limit = int(request.args.get('limit', 50))
        limit = min(max(1, limit), 1000)  # Ensure limit is between 1 and 1000
    except ValueError:
        limit = 50
    
    # Get trades data
    response = databento_advanced.get_trades(symbol, dataset, date, limit)
    return jsonify(response)

@market_bp.route('/databento/advanced/quotes/<symbol>', methods=['GET'])
def get_advanced_quotes(symbol):
    """
    Get quotes (NBBO) data for a symbol using advanced Databento client
    
    Args:
        symbol (str): The stock symbol
        
    Returns:
        JSON: Quotes data
    """
    symbol = symbol.upper().strip()
    
    # Get dataset parameter
    dataset = request.args.get('dataset', 'XNAS_ITCH')
    
    # Get date parameter
    date = request.args.get('date')
    
    # Get limit parameter
    try:
        limit = int(request.args.get('limit', 50))
        limit = min(max(1, limit), 1000)  # Ensure limit is between 1 and 1000
    except ValueError:
        limit = 50
    
    # Get quotes data
    response = databento_advanced.get_quotes(symbol, dataset, date, limit)
    return jsonify(response)

@market_bp.route('/databento/advanced/bars/<symbol>', methods=['GET'])
def get_advanced_bars(symbol):
    """
    Get bar (OHLCV) data for a symbol using advanced Databento client
    
    Args:
        symbol (str): The stock symbol
        
    Returns:
        JSON: Bar data
    """
    symbol = symbol.upper().strip()
    
    # Get dataset parameter
    dataset = request.args.get('dataset', 'XNAS_ITCH')
    
    # Get date parameter
    date = request.args.get('date')
    
    # Get interval parameter
    interval = request.args.get('interval', 'minute')
    if interval not in ['minute', 'hour', 'day']:
        interval = 'minute'
    
    # Get limit parameter
    try:
        limit = int(request.args.get('limit', 50))
        limit = min(max(1, limit), 500)  # Ensure limit is between 1 and 500
    except ValueError:
        limit = 50
    
    # Get bar data
    response = databento_advanced.get_bars(symbol, dataset, interval, date, limit)
    return jsonify(response)

@market_bp.route('/databento/advanced/tape/<symbol>', methods=['GET'])
def get_advanced_tape(symbol):
    """
    Get tape reading data for a symbol using advanced Databento client
    
    Args:
        symbol (str): The stock symbol
        
    Returns:
        JSON: Tape reading data
    """
    symbol = symbol.upper().strip()
    
    # Get dataset parameter
    dataset = request.args.get('dataset', 'XNAS_ITCH')
    
    # Get date parameter
    date = request.args.get('date')
    
    # Get limit parameter
    try:
        limit = int(request.args.get('limit', 100))
        limit = min(max(1, limit), 1000)  # Ensure limit is between 1 and 1000
    except ValueError:
        limit = 100
    
    # Get tape reading data
    response = databento_advanced.get_tape_reading_data(symbol, dataset, date, limit)
    return jsonify(response)

@market_bp.route('/databento/advanced/download/<symbol>', methods=['POST'])
def download_historical_data(symbol):
    """
    Download historical data for a symbol to a local file
    
    Args:
        symbol (str): The stock symbol
        
    Returns:
        JSON: Download status
    """
    symbol = symbol.upper().strip()
    data = request.get_json() or {}
    
    # Get parameters
    dataset = data.get('dataset', 'XNAS_ITCH')
    schema = data.get('schema', 'TRADES')
    start_date = data.get('start_date', (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"))
    end_date = data.get('end_date', datetime.now().strftime("%Y-%m-%d"))
    date_range = (start_date, end_date)
    output_path = data.get('output_path', './data')
    
    # Download historical data
    response = databento_advanced.download_historical_data(
        symbol=symbol,
        dataset=dataset,
        schema=schema,
        date_range=date_range,
        output_path=output_path
    )
    return jsonify(response)

# WebSocket Real-time Market Data Routes

@market_bp.route('/websocket/simulate', methods=['POST'])
def websocket_simulate():
    """
    Enable or disable simulated market data mode
    
    Returns:
        JSON: Simulation status
    """
    import asyncio
    global websocket_client
    
    data = request.json or {}
    enable = data.get('enable', True)
    
    if websocket_client is None:
        from databento_websocket import DatabentoWebSocket
        websocket_client = DatabentoWebSocket()
    
    # Enable or disable simulated mode
    websocket_client.enable_simulated_mode(enable)
    
    # If enabling simulation, connect to start the simulation
    if enable:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(websocket_client.connect())
        finally:
            loop.close()
    
    return jsonify({
        'success': True,
        'simulated_mode': enable,
        'message': f"Simulated mode {'enabled' if enable else 'disabled'}"
    })

@market_bp.route('/websocket/status', methods=['GET'])
def websocket_status():
    """
    Get status of the WebSocket connection
    
    Returns:
        JSON: Status information
    """
    global websocket_client, websocket_task
    
    is_running = websocket_client is not None and websocket_task is not None
    
    active_subscriptions = []
    simulated_mode = False
    
    if websocket_client is not None:
        active_subscriptions = websocket_client.get_active_subscriptions()
        simulated_mode = getattr(websocket_client, 'simulated_mode', False)
    
    # Get data source information from factory
    data_source_info = get_data_source_info()
    
    return jsonify({
        "success": True,
        "connected": is_running,
        "simulated_mode": simulated_mode,
        "active_subscriptions": active_subscriptions,
        "subscription_count": len(active_subscriptions),
        "data_source": {
            "realtime_enabled": data_source_info["realtime_enabled"],
            "using_real_data": data_source_info["using_real_data"],
            "api_key_available": data_source_info["api_key_available"],
            "force_real_data": REALTIME_ENABLED
        }
    })

@market_bp.route('/websocket/connect', methods=['POST'])
def websocket_connect():
    """
    Connect to the Databento WebSocket API
    
    Returns:
        JSON: Connection status
    """
    import asyncio
    global websocket_client, websocket_task
    
    # Don't reconnect if already connected
    if websocket_client is not None and websocket_client.is_connected():
        return jsonify({
            "success": True,
            "message": "WebSocket already connected",
            "already_connected": True,
            "simulated": getattr(websocket_client, 'simulated_mode', False)
        })
    
    # Get request parameters
    data = request.get_json() or {}
    force_simulation = data.get('force_simulation', False)
    use_real_data = data.get('use_real_data', False) or REALTIME_ENABLED
    
    # Create a market data feed using our factory
    try:
        # If forcing simulation despite REALTIME_ENABLED, use a direct DatabentoWebSocket instance
        if force_simulation:
            logger.info("Creating simulated WebSocket connection as explicitly requested")
            websocket_client = DatabentoWebSocket()
            websocket_client.enable_simulated_mode(True)
        else:
            # Use the factory pattern to create the right type of feed based on configuration
            from data_providers.real_data import DatabentoFeed
            from data_providers.simulated_data import SimulatedFeed
            
            if use_real_data and os.environ.get("DATABENTO_API_KEY"):
                logger.info("Creating real data WebSocket connection based on REALTIME_ENABLED=True")
                # Use the real data provider directly to avoid circular imports with the factory
                data_feed = DatabentoFeed()
                websocket_client = data_feed.client
            else:
                if use_real_data:
                    logger.warning("Real data requested but DATABENTO_API_KEY not found; falling back to simulation")
                else:
                    logger.info("Creating simulated WebSocket connection")
                data_feed = SimulatedFeed()
                websocket_client = data_feed.client
    except Exception as e:
        logger.error(f"Error creating market data feed: {str(e)}")
        # Fall back to direct WebSocket client in simulation mode
        websocket_client = DatabentoWebSocket()
        websocket_client.enable_simulated_mode(True)
    
    # If user requested simulation mode, enable it
    if force_simulation:
        logger.info("Using simulated mode due to explicit user request")
        websocket_client.simulated_mode = True
        websocket_client._start_simulation()
        websocket_client.connected = True
        websocket_task = "simulated_connection_active"
        
        return jsonify({
            "success": True,
            "message": "Connected to WebSocket in simulated mode (forced)",
            "simulated": True,
            "connected": True
        })
        
    # Check if API key is available
    if os.environ.get("DATABENTO_API_KEY") is None:
        logger.warning("No Databento API key found, using simulated mode")
        websocket_client.simulated_mode = True
        websocket_client._start_simulation()
        websocket_client.connected = True
        websocket_task = "simulated_connection_active"
        
        return jsonify({
            "success": True,
            "message": "Connected to WebSocket in simulated mode (no API key)",
            "simulated": True,
            "connected": True
        })
    
    # Connect to WebSocket asynchronously using real API
    async def connect_task():
        return await websocket_client.connect()
    
    # Create and run the connection task
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Try to connect to real websocket
        connected = loop.run_until_complete(connect_task())
        if connected:
            # Successfully connected to real API
            logger.info("Successfully connected to Databento WebSocket API")
            return jsonify({
                "success": True,
                "message": "Connected to Databento WebSocket API with live data",
                "simulated": False,
                "connected": True
            })
        else:
            # If connection to real API fails, use simulated mode
            logger.warning("Using simulated WebSocket mode as connection to Databento API failed")
            # Set a flag to indicate simulated mode
            websocket_client.simulated_mode = True
            
            # Explicitly start the simulation
            websocket_client._start_simulation()
            websocket_client.connected = True
            
            # Create a placeholder task so the status check knows we're running
            websocket_task = "simulated_connection_active"
            
            # Return success for simulated mode
            return jsonify({
                "success": True,
                "message": "Connected to WebSocket in simulated mode (fallback)",
                "simulated": True,
                "connected": True
            })
    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}")
        # If any error, use simulated mode
        if websocket_client:
            websocket_client.simulated_mode = True
            
            # Explicitly start the simulation
            websocket_client._start_simulation()
            websocket_client.connected = True
            
            # Create a placeholder task so the status check knows we're running
            websocket_task = "simulated_connection_active"
            
        return jsonify({
            "success": True,
            "message": "Connected to WebSocket in simulated mode",
            "simulated": True,
            "connected": True,
            "error": str(e)
        })
    
    return jsonify({
        "success": True,
        "message": "WebSocket connection initiated"
    })

@market_bp.route('/websocket/disconnect', methods=['POST'])
def websocket_disconnect():
    """
    Disconnect from the Databento WebSocket API
    
    Returns:
        JSON: Disconnection status
    """
    global websocket_client, websocket_task
    
    if websocket_client is None:
        return jsonify({
            "success": False,
            "error": "WebSocket not connected"
        })
    
    # Disconnect client
    import asyncio
    
    if websocket_client is not None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(websocket_client.disconnect())
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Error disconnecting WebSocket: {str(e)}",
                "error": str(e)
            })
    
    # Clear references
    websocket_client = None
    websocket_task = None
    
    return jsonify({
        "success": True,
        "message": "WebSocket disconnected"
    })

@market_bp.route('/websocket/subscribe/<symbol>', methods=['POST'])
def websocket_subscribe(symbol):
    """
    Subscribe to real-time data for a symbol
    
    Args:
        symbol (str): The stock symbol
        
    Returns:
        JSON: Subscription status
    """
    import asyncio
    global websocket_client
    
    symbol = symbol.upper().strip()
    
    if websocket_client is None:
        return jsonify({
            "success": False,
            "error": "WebSocket not connected"
        })
    
    # Get parameters
    data = request.get_json() or {}
    dataset = data.get('dataset', 'XNAS.ITCH')
    data_types = data.get('data_types', ['level2', 'trades', 'quotes'])
    
    # Validate data types
    valid_types = ['level2', 'trades', 'quotes']
    data_types = [dt for dt in data_types if dt in valid_types]
    
    if not data_types:
        return jsonify({
            "success": False,
            "error": "No valid data types specified"
        })
    
    # Add subscriptions
    results = {}
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Register our handlers
    from websocket_handlers import handle_level2_data, handle_trade_data, handle_quote_data
    
    # Subscribe synchronously (the underlying WebSocket methods are async but our wrapper isn't)
    if 'level2' in data_types:
        results['level2'] = websocket_client.subscribe_level2(symbol, handle_level2_data, dataset)
    
    if 'trades' in data_types:
        results['trades'] = websocket_client.subscribe_trades(symbol, handle_trade_data, dataset)
    
    if 'quotes' in data_types:
        results['quotes'] = websocket_client.subscribe_quotes(symbol, handle_quote_data, dataset)
    
    return jsonify({
        "success": True,
        "symbol": symbol,
        "dataset": dataset,
        "data_types": data_types,
        "results": results
    })

@market_bp.route('/websocket/unsubscribe/<symbol>', methods=['POST'])
def websocket_unsubscribe(symbol):
    """
    Unsubscribe from real-time data for a symbol
    
    Args:
        symbol (str): The stock symbol
        
    Returns:
        JSON: Unsubscription status
    """
    import asyncio
    global websocket_client
    
    symbol = symbol.upper().strip()
    
    if websocket_client is None:
        return jsonify({
            "success": False,
            "error": "WebSocket not connected"
        })
    
    # Get parameters
    data = request.get_json() or {}
    dataset = data.get('dataset', 'XNAS.ITCH')
    data_type = data.get('data_type')  # Optional: specific data type to unsubscribe from
    
    # Unsubscribe - this is a synchronous method in our implementation
    success = websocket_client.unsubscribe(symbol, data_type)
    
    return jsonify({
        "success": success,
        "symbol": symbol,
        "message": f"Unsubscribed from {symbol}" if success else f"Failed to unsubscribe from {symbol}"
    })

# WebSocket testing route
@market_bp.route('/websocket/test/<symbol>', methods=['GET'])
def test_websocket(symbol):
    """
    Simple test of WebSocket functionality for a symbol
    
    Args:
        symbol (str): The stock symbol
        
    Returns:
        JSON: Test status
    """
    symbol = symbol.upper().strip()
    
    # Sample code showing how to use the WebSocket client
    test_code = f"""
# WebSocket client example for {symbol}
import asyncio
from databento_websocket import DatabentoWebSocket

async def process_level2(data):
    print(f"📊 Level 2 Update for {symbol}: {{data}}")

async def process_trades(data):
    print(f"💹 Trade for {symbol}: {{data}}")

async def process_quotes(data):
    print(f"💬 Quote for {symbol}: {{data}}")

async def main():
    client = DatabentoWebSocket()
    
    # Enable simulated mode for testing without API key
    # client.enable_simulated_mode(True)
    
    if await client.connect():
        # Subscribe to data feeds with async callbacks
        client.subscribe_level2("{symbol}", process_level2)
        client.subscribe_trades("{symbol}", process_trades)
        client.subscribe_quotes("{symbol}", process_quotes)
        
        try:
            # Listen for 60 seconds
            print("Listening for 60 seconds...")
            await asyncio.sleep(60)
        finally:
            await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
"""
    
    return jsonify({
        "success": True,
        "symbol": symbol,
        "message": "WebSocket test code generated",
        "test_code": test_code
    })
    
# Data Source Information Route
@market_bp.route('/data-source', methods=['GET'])
def get_data_source():
    """
    Get information about the current market data source
    
    Returns:
        JSON: Data source information
    """
    data_source_info = get_data_source_info()
    
    return jsonify({
        "success": True,
        "data_source": {
            "realtime_enabled": data_source_info["realtime_enabled"],
            "using_real_data": data_source_info["using_real_data"],
            "api_key_available": data_source_info["api_key_available"],
            "force_real_data": REALTIME_ENABLED
        }
    })

# Trading Intelligence Routes

@market_bp.route('/realtime/tape/<symbol>', methods=['GET'])
def get_realtime_tape(symbol):
    """
    Get real-time tape reading data from WebSocket data
    Includes Level 2, trades, quotes, and automated insights
    
    Args:
        symbol (str): The stock symbol
        
    Returns:
        JSON: Comprehensive tape reading data with insights
    """
    symbol = symbol.upper().strip()
    
    tape_data = get_tape_reading_data(symbol)
    
    return jsonify({
        "success": True,
        "symbol": symbol,
        "data": tape_data
    })

@market_bp.route('/realtime/insights/<symbol>', methods=['GET'])
def get_realtime_insights(symbol):
    """
    Get automated trading insights based on real-time market data
    
    Args:
        symbol (str): The stock symbol
        
    Returns:
        JSON: Trading insights and recommendations
    """
    symbol = symbol.upper().strip()
    
    insights = get_combined_insights(symbol)
    
    return jsonify({
        "success": True,
        "symbol": symbol,
        "insights": insights
    })