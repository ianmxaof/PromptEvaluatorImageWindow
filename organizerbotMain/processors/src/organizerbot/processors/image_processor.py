"""
Image processing functionality for OrganizerBot
"""
from PIL import Image
from organizerbot.processors.watermark import remove_watermark
from organizerbot.utils.logger import log_action

def process_image(file_path: str) -> None:
    """
    Process an image file with enabled features
    
    Args:
        file_path: Path to the image file
    """
    try:
        with Image.open(file_path) as img:
            # Apply watermark removal if enabled
            if remove_watermark(img):
                log_action(f"Watermark removed from {file_path}")
            
            # Save the processed image
            img.save(file_path)
            log_action(f"Processed image saved: {file_path}")
            
    except Exception as e:
        log_action(f"Error processing image {file_path}: {str(e)}") 