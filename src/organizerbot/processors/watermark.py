"""
Watermark removal for NSFW content
"""
import cv2
import numpy as np
from PIL import Image
import os
from organizerbot.utils.logger import log_action

def remove_watermark(image_path: str) -> None:
    """
    Remove watermarks from NSFW images using advanced image processing
    
    Args:
        image_path: Path to the image file
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to detect text/watermarks
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours of potential watermarks
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by size and position
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter based on typical watermark characteristics
            if w > 50 and h > 10 and y > img.shape[0] * 0.7:
                # Inpaint the watermark area
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        
        # Save the processed image
        cv2.imwrite(image_path, img)
        log_action(f"Watermark removed from {image_path}")
        
    except Exception as e:
        log_action(f"Error removing watermark: {str(e)}")

def add_watermark(image_path: str, text: str) -> None:
    """
    Add a watermark to an image (for testing purposes)
    
    Args:
        image_path: Path to the image file
        text: Watermark text
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
        
        # Add watermark
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        color = (255, 255, 255)
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Position text at bottom right
        x = img.shape[1] - text_width - 10
        y = img.shape[0] - 10
        
        # Add semi-transparent background
        overlay = img.copy()
        cv2.rectangle(
            overlay, (x-5, y-text_height-5),
            (x+text_width+5, y+5), (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        
        # Add text
        cv2.putText(img, text, (x, y), font, font_scale, color, thickness)
        
        # Save the watermarked image
        cv2.imwrite(image_path, img)
        log_action(f"Watermark added to {image_path}")
        
    except Exception as e:
        log_action(f"Error adding watermark: {str(e)}") 