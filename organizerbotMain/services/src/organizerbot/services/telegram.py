"""
Enhanced Telegram integration for PowerCoreAi
"""
import os
import asyncio
from typing import Dict, Any, List
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from organizerbot.utils.logger import log_action
from organizerbot.core.config import load_config, save_config
from organizerbot.processors.image_processor import process_image
from organizerbot.processors.categorizer import categorize_image

class TelegramBot:
    """Enhanced Telegram bot with command handling"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bot = Bot(token=config["bot_token"])
        self.app = Application.builder().token(config["bot_token"]).build()
        self.setup_handlers()

    def setup_handlers(self):
        """Setup command handlers"""
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("status", self.status_command))
        self.app.add_handler(CommandHandler("folders", self.list_folders_command))
        self.app.add_handler(CommandHandler("add_folder", self.add_folder_command))
        self.app.add_handler(CommandHandler("remove_folder", self.remove_folder_command))
        self.app.add_handler(CommandHandler("set_source", self.set_source_command))
        self.app.add_handler(CommandHandler("list_sources", self.list_sources_command))
        self.app.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        await update.message.reply_text(
            "👋 Welcome to PowerCoreAi!\n\n"
            "I can help you organize and process your images. Here are some commands:\n"
            "/help - Show this help message\n"
            "/process <image> - Process an image\n"
            "/categorize <image> - Categorize an image\n"
            "/settings - View your settings"
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = (
            "📚 Available Commands:\n\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/status - Show current settings\n"
            "/folders - List watched folders\n"
            "/add_folder <path> - Add a folder to watch\n"
            "/remove_folder <path> - Remove a watched folder\n"
            "/set_source <path> - Set a source folder for images\n"
            "/list_sources - List all source folders\n"
            "📸 Send a photo to process and categorize it"
        )
        await update.message.reply_text(help_text)

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        status = (
            f"⚙️ Current Settings:\n\n"
            f"Watermark Removal: {'✅' if self.config['features']['watermark_removal'] else '❌'}\n"
            f"Enhancement: {'✅' if self.config['features']['enhancement'] else '❌'}\n"
            f"Auto-Upload: {'✅' if self.config['features']['auto_upload'] else '❌'}\n"
            f"Watched Folders: {len(self.config['watch_folders'])}\n"
            f"Source Folders: {len(self.config.get('source_folders', []))}"
        )
        await update.message.reply_text(status)

    async def list_folders_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /folders command"""
        folders = "\n".join(f"📁 {folder}" for folder in self.config['watch_folders'])
        await update.message.reply_text(f"📂 Watched Folders:\n\n{folders}")

    async def add_folder_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /add_folder command"""
        if not context.args:
            await update.message.reply_text("Please provide a folder path")
            return

        folder = context.args[0]
        if os.path.isdir(folder):
            if folder not in self.config['watch_folders']:
                self.config['watch_folders'].append(folder)
                save_config(self.config)
                await update.message.reply_text(f"✅ Added folder: {folder}")
            else:
                await update.message.reply_text("⚠️ Folder already in watch list")
        else:
            await update.message.reply_text("❌ Invalid folder path")

    async def remove_folder_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /remove_folder command"""
        if not context.args:
            await update.message.reply_text("Please provide a folder path")
            return

        folder = context.args[0]
        if folder in self.config['watch_folders']:
            self.config['watch_folders'].remove(folder)
            save_config(self.config)
            await update.message.reply_text(f"✅ Removed folder: {folder}")
        else:
            await update.message.reply_text("❌ Folder not found in watch list")

    async def set_source_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /set_source command"""
        if not context.args:
            await update.message.reply_text("Please provide a source folder path")
            return

        folder = context.args[0]
        if os.path.isdir(folder):
            if 'source_folders' not in self.config:
                self.config['source_folders'] = []
            
            if folder not in self.config['source_folders']:
                self.config['source_folders'].append(folder)
                save_config(self.config)
                await update.message.reply_text(f"✅ Set source folder: {folder}")
            else:
                await update.message.reply_text("⚠️ Folder already set as source")
        else:
            await update.message.reply_text("❌ Invalid folder path")

    async def list_sources_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /list_sources command"""
        sources = self.config.get('source_folders', [])
        if not sources:
            await update.message.reply_text("No source folders configured")
            return

        sources_text = "\n".join(f"📁 {source}" for source in sources)
        await update.message.reply_text(f"📂 Source Folders:\n\n{sources_text}")

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming photos"""
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        
        # Download the photo
        temp_path = f"temp_{photo.file_id}.jpg"
        await file.download_to_drive(temp_path)
        
        # Process and categorize
        process_image(temp_path)
        category = categorize_image(temp_path)
        
        # Get source folders
        source_folders = self.config.get('source_folders', [])
        if source_folders:
            # Move to appropriate category folder in first source folder
            target_folder = os.path.join(source_folders[0], category)
            os.makedirs(target_folder, exist_ok=True)
            target_path = os.path.join(target_folder, os.path.basename(temp_path))
            os.rename(temp_path, target_path)
            
            await update.message.reply_text(
                f"📸 Image processed and moved!\n"
                f"Category: {category}\n"
                f"Location: {target_path}"
            )
        else:
            # No source folders configured
            await update.message.reply_text(
                f"📸 Image processed!\n"
                f"Category: {category}\n"
                f"⚠️ No source folders configured. Use /set_source to add one."
            )
            os.remove(temp_path)

    async def start(self):
        """Start the bot"""
        await self.app.initialize()
        await self.app.start()
        await self.app.run_polling()

def start_telegram_bot(config: Dict[str, Any]) -> None:
    """Start the Telegram bot"""
    bot = TelegramBot(config)
    asyncio.run(bot.start()) 