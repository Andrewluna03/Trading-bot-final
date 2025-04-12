import base64
import io
import logging
from PIL import ImageGrab
from PIL import Image

logger = logging.getLogger(__name__)

def capture_screen():
    """
    Capture the current screen and return it as a base64 encoded string.
    
    Returns:
        str: Base64 encoded screenshot
    """
    try:
        # Capture the screen
        screenshot = ImageGrab.grab()
        
        # Save to a bytes buffer
        buffer = io.BytesIO()
        screenshot.save(buffer, format="JPEG")
        buffer.seek(0)
        
        # Encode to base64
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        logger.debug("Screen captured successfully")
        return img_str
    except Exception as e:
        logger.error(f"Error capturing screen: {str(e)}")
        raise Exception(f"Failed to capture screen: {str(e)}")
