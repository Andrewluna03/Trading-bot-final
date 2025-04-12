import os
import logging
import re
import base64
from flask import render_template, request, jsonify
from app import app, db
from models import ScreenAnalysis
from screen_capture import capture_screen
from ocr_processor import extract_text_from_image
from openai_analyzer import analyze_screenshot

# Set up logging
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Render the main page of the application."""
    # Check if OpenAI API key is available and valid
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    return render_template('index.html', openai_api_key=openai_api_key)
    
@app.route('/websocket-test')
def websocket_test():
    """Render the WebSocket testing interface"""
    return render_template('websocket_test.html')
    
@app.route('/dashboard')
def dashboard():
    """Render the trading dashboard with real-time data"""
    # Set up popular stock symbols for the dashboard
    default_symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    
    # Get symbols from query string, otherwise use defaults
    symbols = request.args.get('symbols', ','.join(default_symbols)).upper().split(',')
    
    # Filter out any empty strings
    symbols = [s.strip() for s in symbols if s.strip()]
    
    # If no valid symbols provided, use defaults
    if not symbols:
        symbols = default_symbols
        
    return render_template('dashboard.html', symbols=symbols)

@app.route('/capture', methods=['POST'])
def process_screenshot():
    """
    Process a screenshot (captured or uploaded), extract text with OCR, 
    and analyze it with OpenAI.
    """
    try:
        # Check if we're using file upload or screen capture
        if 'screenshot' in request.files:
            # Process uploaded file
            file = request.files['screenshot']
            if file:
                # Read the image data
                img_data = file.read()
                # Convert to base64
                screenshot_data = base64.b64encode(img_data).decode('utf-8')
            else:
                raise ValueError("No file provided")
        else:
            # Capture the screen
            screenshot_data = capture_screen()
        
        # Get platform type and analysis depth from form data
        platform_type = request.form.get('platform_type', 'generic')
        analysis_depth = request.form.get('analysis_depth', 'standard')
        
        # Extract text using OCR
        extracted_text = extract_text_from_image(screenshot_data)
        
        # Analyze with OpenAI or fallback, passing platform information
        analysis = analyze_screenshot(
            screenshot_data, 
            extracted_text, 
            platform_type=platform_type,
            analysis_depth=analysis_depth
        )
        
        # Parse trading information from analysis
        symbol = extract_trading_info(analysis, 'symbol')
        bid_price = extract_trading_info(analysis, 'bid', float)
        ask_price = extract_trading_info(analysis, 'ask', float)
        spread = extract_trading_info(analysis, 'spread', float)
        recommendation = extract_recommendation(analysis)
        
        # Set default recommendation if none detected
        if not recommendation:
            recommendation = "Wait"
        
        # Save to database
        screen_analysis = ScreenAnalysis(
            extracted_text=extracted_text,
            analysis_result=analysis,
            symbol=symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            spread=spread,
            recommendation=recommendation
        )
        
        db.session.add(screen_analysis)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'extracted_text': extracted_text,
            'analysis': analysis,
            'trading_info': {
                'symbol': symbol,
                'bid_price': bid_price,
                'ask_price': ask_price,
                'spread': spread,
                'recommendation': recommendation
            }
        })
    except Exception as e:
        logger.error(f"Error processing screenshot: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/analysis-history', methods=['GET'])
def get_analysis_history():
    """Get the history of trading screen analyses"""
    try:
        # Get last 20 entries, newest first
        history = ScreenAnalysis.query.order_by(ScreenAnalysis.timestamp.desc()).limit(20).all()
        history_list = [record.to_dict() for record in history]
        
        return jsonify({
            'success': True,
            'history': history_list
        })
    except Exception as e:
        logger.error(f"Error retrieving analysis history: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Helper functions for parsing analysis results
def extract_trading_info(analysis, field_name, conv_func=None):
    """Extract trading information from analysis text"""
    if conv_func is None:
        conv_func = str
        
    patterns = {
        'symbol': r'(?:Symbol|Ticker):\s*([A-Z]+)',
        'bid': r'(?:Bid|Bid Price):\s*\$?(\d+\.?\d*)',
        'ask': r'(?:Ask|Ask Price|Offer):\s*\$?(\d+\.?\d*)',
        'spread': r'(?:Spread):\s*\$?(\d+\.?\d*)'
    }
    
    if field_name in patterns:
        match = re.search(patterns[field_name], analysis, re.IGNORECASE)
        if match:
            try:
                return conv_func(match.group(1))
            except (ValueError, TypeError):
                return None
    return None

def extract_recommendation(analysis):
    """Extract trading recommendation from analysis text"""
    patterns = [
        r'(?:Recommendation|Trading Signal|Signal|Action):\s*(Enter|Buy|Long|Exit|Sell|Short|Hold|Wait)',
        r'(?:I recommend to|You should|Best action would be to)\s*(enter|buy|long|exit|sell|short|hold|wait)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, analysis, re.IGNORECASE)
        if match:
            action = match.group(1).lower()
            if action in ('enter', 'buy', 'long'):
                return 'Enter'
            elif action in ('exit', 'sell', 'short'):
                return 'Exit'
            elif action == 'hold':
                return 'Hold'
            elif action == 'wait':
                return 'Wait'
    
    return None