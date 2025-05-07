"""
Image processing functionality for OrganizerBot
"""
import cv2
import numpy as np
from PIL import Image
from organizerbot.utils.logger import log_action

def process_image(image_path: str) -> None:
    """
    Process an image with various enhancements
    
    Args:
        image_path: Path to the image file
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
        
        # Convert to LAB color space for better color enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge channels
        limg = cv2.merge((cl,a,b))
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Save the processed image
        cv2.imwrite(image_path, enhanced)
        log_action(f"Image enhanced: {image_path}")
        
    except Exception as e:
        log_action(f"Error processing image: {str(e)}") 