"""
File system monitoring for OrganizerBot
"""
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from organizerbot.processors.image_processor import process_image
from organizerbot.services.telegram import upload_to_telegram
from organizerbot.utils.logger import log_action

class FileEventHandler(FileSystemEventHandler):
    """Handle file system events"""
    def __init__(self, config):
        self.config = config
        super().__init__()

    def on_modified(self, event):
        if event.is_directory:
            return
        
        file_path = event.src_path
        log_action(f"File detected: {file_path}")
        
        # Process image if enhancement is enabled
        if self.config.features["enhancement"]:
            process_image(file_path)
        
        # Upload to Telegram if enabled
        if self.config.features["auto_upload"]:
            upload_to_telegram(file_path, self.config.telegram_config)
        
        log_action(f"Processed {file_path}")

def start_file_watcher(directory: str) -> None:
    """Start monitoring the specified directory"""
    from organizerbot.core.config import load_config
    
    config = load_config()
    event_handler = FileEventHandler(config)
    observer = Observer()
    
    observer.schedule(event_handler, directory, recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join() 