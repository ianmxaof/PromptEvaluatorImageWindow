import os
import shutil
import json
from categorizer.model import categorize_media
from enhancer_tools import enhance_image
from datetime import datetime
from PIL import Image, ImageEnhance

def enhance_image(input_path, output_path):
    image = Image.open(input_path)
    
    # Enhance sharpness, contrast, etc. – tune to taste
    sharpness = ImageEnhance.Sharpness(image).enhance(2.0)
    contrast = ImageEnhance.Contrast(sharpness).enhance(1.5)
    brightness = ImageEnhance.Brightness(contrast).enhance(1.2)

    brightness.save(output_path)

MEDIA_DIR = "media/"
ORGANIZED_DIR = "organized/"
TOPICS_FILE = "topics.json"
LOG_FILE = "logs/organizer.log"

# Load topics from JSON
def load_topics():
    with open(TOPICS_FILE, 'r') as f:
        return json.load(f)

# Organize media by topic
def organize_by_topic():
    topics = load_topics()

    for filename in os.listdir(MEDIA_DIR):
        filepath = os.path.join(MEDIA_DIR, filename)
        if filename.endswith(".webm"):
            continue

        category = categorize_media(filepath, topics)  # Categorize via AI
        category_folder = os.path.join(ORGANIZED_DIR, category)

        os.makedirs(category_folder, exist_ok=True)
        shutil.move(filepath, os.path.join(category_folder, filename))

        # Check if enhancement or watermark removal is needed
        if needs_enhancement(filepath):
            enhance_image(filepath)
        if needs_watermark(filepath):
            remove_watermark(filepath)

        log_action(f"Moved {filename} to {category}.")

# Log actions
def log_action(message):
    with open(LOG_FILE, 'a') as log:
        log.write(f"{message}\n")
    print(message)

# Enhancement check (stub)
def needs_enhancement(filepath):
    # Add logic to determine if enhancement is needed (e.g., based on file size, quality, etc.)
    return True

# Watermark check (stub)
def needs_watermark(filepath):
    # Add logic to check if watermark removal is required (e.g., based on image metadata)
    return True
