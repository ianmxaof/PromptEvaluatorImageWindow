import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from processor import organize_by_topic
from processor import log_action
from telegram_uploader import upload_file

class Watcher(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        print(f"File detected: {event.src_path}")
        # Check for toggles before processing
        if upload_enabled:
            upload_file(event.src_path)
        organize_by_topic()  # Process the file
        log_action(f"Processed {event.src_path}")

def start_watching(directory):
    event_handler = Watcher()
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
