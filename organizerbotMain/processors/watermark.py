from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import os

def add_watermark(input_path, output_path, text="© OrganizerBot", font_size=36):
    try:
        image = Image.open(input_path).convert("RGBA")

        # Create a watermark layer
        watermark = Image.new("RGBA", image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(watermark)

        # Load default font
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # Calculate text position (bottom right)
        text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
        x = image.width - text_width - 10
        y = image.height - text_height - 10

        logo = Image.open("logo.png").convert("RGBA")
        logo = logo.resize((100, 30))  # Resize as needed

        # Paste in bottom-right corner
        image.paste(logo, (image.width - logo.width - 10, image.height - logo.height - 10), mask=logo)


        # Merge watermark with original image
        watermarked = Image.alpha_composite(image, watermark)
        watermarked = watermarked.convert("RGB")  # Drop alpha for saving as JPEG/PNG

        # Save to output
        watermarked.save(output_path)
        print("✅ Watermark added.")
    except Exception as e:
        print(f"❌ Error in adding watermark: {e}")

def enhance_image(input_path, output_path, brightness=1.2, contrast=1.3):
    try:
        image = Image.open(input_path)

        # Apply brightness
        enhancer_bright = ImageEnhance.Brightness(image)
        image = enhancer_bright.enhance(brightness)

        # Apply contrast
        enhancer_contrast = ImageEnhance.Contrast(image)
        image = enhancer_contrast.enhance(contrast)

        image.save(output_path)
        print("✅ Image enhancement complete.")
    except Exception as e:
        print(f"❌ Error in enhancing image: {e}")
