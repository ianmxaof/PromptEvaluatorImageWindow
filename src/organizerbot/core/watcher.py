"""
File system monitoring for OrganizerBot
"""
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from organizerbot.processors.image_processor import process_image
from organizerbot.processors.categorizer import categorize_image
from organizerbot.utils.logger import log_action
import os
import shutil

class FileEventHandler(FileSystemEventHandler):
    """Handle file system events"""
    def __init__(self, config):
        self.config = config
        self.processing_files = set()  # Track files being processed
        super().__init__()

    def on_created(self, event):
        """Handle file creation events"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        if file_path in self.processing_files:
            return
            
        self.processing_files.add(file_path)
        try:
            self._process_file(file_path)
        finally:
            self.processing_files.remove(file_path)

    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        if file_path in self.processing_files:
            return
            
        self.processing_files.add(file_path)
        try:
            self._process_file(file_path)
        finally:
            self.processing_files.remove(file_path)

    def _process_file(self, file_path: str):
        """Process a single file"""
        try:
            # Check if file exists and is an image
            if not os.path.exists(file_path):
                return
                
            if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                return
                
            log_action(f"Processing file: {file_path}")
            
            # Process image if enhancement is enabled
            if self.config.features.get("enhancement", False):
                process_image(file_path)
            
            # Categorize the image
            category = categorize_image(file_path)
            log_action(f"Image categorized as: {category}")
            
            # Move to appropriate category folder
            if self.config.source_folders:
                target_folder = os.path.join(self.config.source_folders[0], category)
                os.makedirs(target_folder, exist_ok=True)
                
                # Generate unique filename if needed
                base_name = os.path.basename(file_path)
                target_path = os.path.join(target_folder, base_name)
                counter = 1
                while os.path.exists(target_path):
                    name, ext = os.path.splitext(base_name)
                    target_path = os.path.join(target_folder, f"{name}_{counter}{ext}")
                    counter += 1
                
                # Move the file
                shutil.move(file_path, target_path)
                log_action(f"Moved to: {target_path}")
                
        except Exception as e:
            log_action(f"Error processing file {file_path}: {str(e)}")

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