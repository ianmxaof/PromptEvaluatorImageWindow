# categorizer/utils.py
import os
import shutil

# Define your categories here
CATEGORIES = [
    "TITS", "ASS", "MILF", "LATINA", "BJ", "CHEATING", "AI", "GIFS", "RANDOM"
]

def categorize_media(media_name):
    """
    Function to categorize media based on its name (can be expanded to other properties like file content)
    """
    for category in CATEGORIES:
        if category.lower() in media_name.lower():
            return category
    return "Uncategorized"

def move_to_category(media_name, category):
    """
    Function to move the file to the appropriate category folder.
    If the category doesn't exist, create it.
    """
    category_folder = os.path.join("categorizer", category)

    # Create category folder if it doesn't exist
    if not os.path.exists(category_folder):
        os.makedirs(category_folder)

    # Move the media file to the category folder
    os.rename(media_name, os.path.join(category_folder, media_name))
    print(f"Moved {media_name} to {category_folder}")
