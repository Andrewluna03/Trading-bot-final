"""
Test script for Databento WebSocket functionality
"""
import asyncio
import json
import logging
import sys
from typing import Dict, Any

from databento_websocket import DatabentoWebSocket

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure your test parameters
TEST_SYMBOL = "AAPL"
TEST_DURATION = 60  # seconds

# Define callback handlers
async def handle_level2_data(data: Dict[str, Any]):
    """
    Process Level 2 market depth data
    
    Args:
        data (Dict): The Level 2 data from the WebSocket
    """
    symbol = data.get('symbol', 'Unknown')
    schema = data.get('schema', 'Unknown')
    
    try:
        # Process MBP-10 data format
        if schema == 'mbp_10':
            bids = []
            asks = []
            
            # Extract bid and ask levels
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
            
            # Print summary
            if bids and asks:
                spread = asks[0]['price'] - bids[0]['price']
                logger.info(f"🔵 {symbol} L2: Top Bid: ${bids[0]['price']} x {bids[0]['size']} | " + 
                          f"Top Ask: ${asks[0]['price']} x {asks[0]['size']} | " + 
                          f"Spread: ${spread:.4f}")
                
                # Check for patterns
                if any(b['size'] > 10000 for b in bids[:3]):
                    logger.info(f"🟢 {symbol}: Large buy orders detected!")
                
                if any(a['size'] > 10000 for a in asks[:3]):
                    logger.info(f"🔴 {symbol}: Large sell orders detected!")
                    
    except Exception as e:
        logger.error(f"Error processing Level 2 data: {e}")

async def handle_trade_data(data: Dict[str, Any]):
    """
    Process trade data
    
    Args:
        data (Dict): The trade data from the WebSocket
    """
    symbol = data.get('symbol', 'Unknown')
    price = data.get('price', 0)
    size = data.get('size', 0)
    side = 'Buy' if data.get('side') == 'B' else 'Sell'
    
    logger.info(f"💰 {symbol} Trade: {side} {size} @ ${price}")
    
    # Check for large trades
    if size > 10000:
        logger.info(f"⚠️ {symbol}: Large {side.lower()} of {size} shares!")

async def handle_quote_data(data: Dict[str, Any]):
    """
    Process quote data
    
    Args:
        data (Dict): The quote data from the WebSocket
    """
    symbol = data.get('symbol', 'Unknown')
    bid_price = data.get('bid_px_00', 0)
    bid_size = data.get('bid_sz_00', 0)
    ask_price = data.get('ask_px_00', 0)
    ask_size = data.get('ask_sz_00', 0)
    
    if bid_price > 0 and ask_price > 0:
        spread = ask_price - bid_price
        logger.info(f"📊 {symbol} Quote: Bid: ${bid_price} x {bid_size} | " + 
                  f"Ask: ${ask_price} x {ask_size} | " + 
                  f"Spread: ${spread:.4f}")

async def main():
    """Main test function"""
    logger.info("Starting Databento WebSocket test")
    
    # Create WebSocket client
    client = DatabentoWebSocket()
    
    # Connect to the WebSocket API
    logger.info("Connecting to Databento WebSocket API...")
    connected = await client.connect()
    
    if not connected:
        logger.error("Failed to connect to Databento WebSocket API")
        return
    
    logger.info(f"Connected successfully. Testing with symbol: {TEST_SYMBOL}")
    
    try:
        # Subscribe to all data types
        client.subscribe_level2(TEST_SYMBOL, handle_level2_data)
        logger.info(f"Subscribed to Level 2 data for {TEST_SYMBOL}")
        
        client.subscribe_trades(TEST_SYMBOL, handle_trade_data)
        logger.info(f"Subscribed to trades data for {TEST_SYMBOL}")
        
        client.subscribe_quotes(TEST_SYMBOL, handle_quote_data)
        logger.info(f"Subscribed to quotes data for {TEST_SYMBOL}")
        
        # Run for the specified duration
        logger.info(f"Running test for {TEST_DURATION} seconds...")
        await asyncio.sleep(TEST_DURATION)
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
        
    finally:
        # Disconnect
        logger.info("Test complete. Disconnecting...")
        await client.disconnect()
        logger.info("Disconnected")

if __name__ == "__main__":
    # Get custom symbol from command line if provided
    if len(sys.argv) > 1:
        TEST_SYMBOL = sys.argv[1].upper()
    
    # Run the test
    asyncio.run(main())