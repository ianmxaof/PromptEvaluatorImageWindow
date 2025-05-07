import sys
import os
import telebot
import logging
from categorizer.utils import categorize_media, move_to_category

# Add project directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)

# Initialize bot
bot = telebot.TeleBot("YOUR_BOT_TOKEN_HERE")  # <- replace with your token

@bot.message_handler(content_types=["photo", "video"])
def handle_media(message):
    try:
        # Get file info
        file_info = bot.get_file(
            message.photo[-1].file_id if message.content_type == "photo" else message.video.file_id
        )
        downloaded_file = bot.download_file(file_info.file_path)

        # Save temp file
        file_name = file_info.file_path.split("/")[-1]
        with open(file_name, 'wb') as new_file:
            new_file.write(downloaded_file)

        logging.info(f"Downloaded: {file_name}")

        # Categorize + move
        category = categorize_media(file_name)
        move_to_category(file_name, category)
        logging.info(f"Categorized '{file_name}' as '{category}'")

        # Clean up
        if os.path.exists(file_name):
            os.remove(file_name)
            logging.info(f"Deleted temp file: {file_name}")

        # User feedback
        bot.reply_to(message, f"✅ Categorized as: {category}")

    except Exception as e:
        logging.error(f"Error handling media: {e}")
        bot.reply_to(message, "⚠️ Something went wrong while processing your file.")

# Run bot
bot.polling()
