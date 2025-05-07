import os
import shutil
import json
from categorizer.model import categorize_media
from enhancer import enhance_image
from watermark import remove_watermark, add_watermark
from datetime import datetime

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
