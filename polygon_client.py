import os
import logging
import requests
from typing import Dict, Any, Optional, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolygonClient:
    """
    Client for the Polygon.io API to fetch real-time market data
    """
    
    def __init__(self):
        """Initialize the Polygon client with API key from environment"""
        self.api_key = os.environ.get('POLYGON_API_KEY')
        self.base_url = 'https://api.polygon.io'
        
        if not self.api_key:
            logger.warning("No Polygon API key found. Real-time data will not be available.")
    
    def get_ticker_details(self, symbol: str) -> Dict[str, Any]:
        """
        Get details for a specific ticker
        
        Args:
            symbol (str): The stock symbol (e.g., AAPL)
            
        Returns:
            Dict: Ticker details or error information
        """
        if not self.api_key:
            return self._error_response("No API key available", "missing_api_key", symbol)
            
        try:
            url = f"{self.base_url}/v3/reference/tickers/{symbol}"
            response = requests.get(url, params={'apiKey': self.api_key})
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                return self._error_response(
                    "API access unauthorized (401). Please provide a valid Polygon API key.",
                    "invalid_api_key",
                    symbol
                )
            else:
                return self._error_response(
                    f"API request failed with status {response.status_code}",
                    "api_error",
                    symbol
                )
                
        except Exception as e:
            logger.error(f"Error fetching ticker details for {symbol}: {str(e)}")
            return self._error_response(str(e), "request_error", symbol)
    
    def get_last_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get the last quote for a symbol (Level 1 data)
        
        Args:
            symbol (str): The stock symbol
            
        Returns:
            Dict: Quote data or error information
        """
        if not self.api_key:
            return self._error_response("No API key available", "missing_api_key", symbol)
            
        try:
            # Instead of using the NBBO endpoint which requires a paid subscription,
            # we'll use the previous close endpoint which is available on the free tier
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/prev"
            response = requests.get(url, params={'apiKey': self.api_key})
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                return self._error_response(
                    "API access unauthorized (401). Please provide a valid Polygon API key.",
                    "invalid_api_key",
                    symbol
                )
            else:
                return self._error_response(
                    f"API request failed with status {response.status_code}",
                    "api_error",
                    symbol
                )
                
        except Exception as e:
            logger.error(f"Error fetching last quote for {symbol}: {str(e)}")
            return self._error_response(str(e), "request_error", symbol)
    
    def get_level2_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get Level 2 order book data for a symbol
        
        Args:
            symbol (str): The stock symbol
            
        Returns:
            Dict: Level 2 data or error information
        """
        if not self.api_key:
            return self._error_response("No API key available", "missing_api_key", symbol)
            
        try:
            # Using daily bars endpoint which is available on free tier
            # Free tier doesn't provide true Level 2 data, so we'll use daily bar data
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/2023-01-01/{self._get_today()}"
            params = {
                'apiKey': self.api_key,
                'limit': 50,
                'sort': 'desc'
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'results': response.json().get('results', []),
                    'ticker': symbol,
                    'note': 'Free tier API - showing historical daily bars instead of true Level 2 data'
                }
            elif response.status_code == 401:
                return self._error_response(
                    "API access unauthorized (401). Please provide a valid Polygon API key.",
                    "invalid_api_key",
                    symbol
                )
            else:
                return self._error_response(
                    f"API request failed with status {response.status_code}",
                    "api_error",
                    symbol
                )
                
        except Exception as e:
            logger.error(f"Error fetching Level 2 data for {symbol}: {str(e)}")
            return self._error_response(str(e), "request_error", symbol)
    
    def get_time_and_sales(self, symbol: str, limit: int = 50) -> Dict[str, Any]:
        """
        Get Time & Sales data (recent trades) for a symbol
        
        Args:
            symbol (str): The stock symbol
            limit (int): Number of recent trades to fetch
            
        Returns:
            Dict: Time & Sales data or error information
        """
        if not self.api_key:
            return self._error_response("No API key available", "missing_api_key", symbol)
            
        try:
            # Use minute bars endpoint which is available on free tier
            # Free tier doesn't provide true Time & Sales data, so we'll use minute bars
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/minute/{self._get_today()}/{self._get_today()}"
            params = {
                'apiKey': self.api_key,
                'limit': limit,
                'sort': 'desc'
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                # Transform the response to match expected format for our application
                data = response.json()
                # Create simulated trades from the minute bars
                results = []
                for bar in data.get('results', []):
                    # Add a simulated trade for each minute bar
                    timestamp = bar.get('t')
                    if timestamp:
                        results.append({
                            't': timestamp,
                            'p': bar.get('c'),  # Close price
                            's': bar.get('v'),  # Volume
                            'x': 'SIM',  # Simulate exchange
                            'i': f"sim_{timestamp}"  # Simulate trade ID
                        })
                        
                return {
                    'success': True,
                    'results': results,
                    'ticker': symbol,
                    'note': 'Free tier API - showing minute bars instead of true Time & Sales data'
                }
            elif response.status_code == 401:
                return self._error_response(
                    "API access unauthorized (401). Please provide a valid Polygon API key.",
                    "invalid_api_key",
                    symbol
                )
            else:
                return self._error_response(
                    f"API request failed with status {response.status_code}",
                    "api_error",
                    symbol
                )
                
        except Exception as e:
            logger.error(f"Error fetching Time & Sales for {symbol}: {str(e)}")
            return self._error_response(str(e), "request_error", symbol)
            
    def _get_today(self) -> str:
        """Get today's date in YYYY-MM-DD format"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d')
    
    def _error_response(self, message: str, error_type: str, symbol: str) -> Dict[str, Any]:
        """Create a standardized error response"""
        return {
            'success': False,
            'error': {
                'message': message,
                'error_type': error_type,
                'symbol': symbol
            }
        }