from PIL import Image, ImageEnhance
import os

def enhance_image(image_path):
    """Basic image enhancement: boost contrast and sharpness."""
    try:
        image = Image.open(image_path)
        enhancer_contrast = ImageEnhance.Contrast(image)
        image = enhancer_contrast.enhance(1.5)

        enhancer_sharpness = ImageEnhance.Sharpness(image)
        image = enhancer_sharpness.enhance(2.0)

        # Save the enhanced image as a temporary file
        base, ext = os.path.splitext(image_path)
        enhanced_path = f"{base}_enhanced{ext}"
        image.save(enhanced_path)

        return enhanced_path
    except Exception as e:
        print(f"Enhancement failed for {image_path}: {e}")
        return image_path
