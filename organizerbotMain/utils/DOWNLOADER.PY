import telebot

API_TOKEN = "YOUR_API_TOKEN"  # Use your bot's API token here
bot = telebot.TeleBot(API_TOKEN)

def download_media(file_id, file_name):
    """Download media files from Telegram."""
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    with open(f"media/{file_name}", 'wb') as new_file:
        new_file.write(downloaded_file)
    print(f"Downloaded {file_name} successfully.")
