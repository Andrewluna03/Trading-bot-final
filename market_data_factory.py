"""
Market data feed factory
"""
import logging
import os
from typing import Optional, Union, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration
REALTIME_ENABLED = True  # Force real data when possible

def create_market_data_feed() -> Any:
    """
    Factory function to create the appropriate market data feed
    based on the REALTIME_ENABLED configuration
    
    Returns:
        Market data feed instance
    """
    # Import directly to avoid circular imports
    if REALTIME_ENABLED:
        try:
            # Check for API key
            if not os.environ.get("DATABENTO_API_KEY"):
                logger.critical("DATABENTO_API_KEY environment variable is not set but REALTIME_ENABLED=True")
                logger.critical("Cannot create real-time data feed without API key")
                raise ValueError("DATABENTO_API_KEY environment variable must be set for real data feed")
                
            from data_providers.real_data import DatabentoFeed
            logger.info("Creating real-time market data feed (REALTIME_ENABLED=True)")
            return DatabentoFeed()
        except Exception as e:
            if os.environ.get("REALTIME_STRICT", "false").lower() == "true":
                # In strict mode, failure to create real data feed is fatal
                logger.critical(f"Failed to create real-time data feed and REALTIME_STRICT=True: {str(e)}")
                raise
            else:
                # In non-strict mode, fall back to simulation
                logger.warning(f"Failed to create real-time data feed, falling back to simulation: {str(e)}")
                from data_providers.simulated_data import SimulatedFeed
                return SimulatedFeed()
    else:
        from data_providers.simulated_data import SimulatedFeed
        logger.info("Creating simulated market data feed (REALTIME_ENABLED=False)")
        return SimulatedFeed()
        
def get_data_source_info() -> Dict[str, Any]:
    """
    Get information about the current data source
    
    Returns:
        Dict with data source information
    """
    return {
        "realtime_enabled": REALTIME_ENABLED,
        "using_real_data": REALTIME_ENABLED and os.environ.get("DATABENTO_API_KEY") is not None,
        "api_key_available": os.environ.get("DATABENTO_API_KEY") is not None,
        "strict_mode": os.environ.get("REALTIME_STRICT", "false").lower() == "true"
    }