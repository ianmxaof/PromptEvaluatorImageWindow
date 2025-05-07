from PIL import Image, ImageEnhance
import os

def enhance_image(image_path):
    """Enhance the quality of an image."""
    try:
        # Open image
        image = Image.open(image_path)
        
        # Optionally, resize the image if it's too large or small
        image = resize_image(image)
        
        # Enhance image sharpness
        image = enhance_sharpness(image)
        
        # Enhance brightness, contrast, or color (optional)
        image = enhance_brightness(image)
        image = enhance_contrast(image)
        
        # Save the enhanced image
        image.save(image_path)
        print(f"Image enhanced: {image_path}")
    except Exception as e:
        print(f"Error enhancing image {image_path}: {str(e)}")

def resize_image(image, target_size=(1024, 1024)):
    """Resize image to a target size while maintaining the aspect ratio."""
    width, height = image.size
    new_width, new_height = target_size
    
    # Calculate aspect ratio
    aspect_ratio = width / height
    if aspect_ratio > 1:
        new_height = int(new_width / aspect_ratio)
    else:
        new_width = int(new_height * aspect_ratio)
    
    return image.resize((new_width, new_height))

def enhance_sharpness(image, factor=2.0):
    """Enhance the sharpness of the image."""
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)

def enhance_brightness(image, factor=1.5):
    """Enhance the brightness of the image."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def enhance_contrast(image, factor=1.5):
    """Enhance the contrast of the image."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)
