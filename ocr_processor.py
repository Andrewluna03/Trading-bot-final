import base64
import io
import logging
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

def extract_text_from_image(base64_image):
    """
    Extract text from a base64 encoded image using OCR.
    
    Args:
        base64_image (str): Base64 encoded image
        
    Returns:
        str: Extracted text from the image
    """
    try:
        # Decode the base64 image
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
        
        # Use pytesseract to extract text
        extracted_text = pytesseract.image_to_string(image)
        
        logger.debug(f"Text extracted: {extracted_text[:100]}...")
        return extracted_text
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return "Error extracting text from image"
