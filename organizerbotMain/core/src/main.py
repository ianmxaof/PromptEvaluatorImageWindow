"""
OrganizerBot - Main entry point
"""
from organizerbot.gui.tray import run_tray_application
from organizerbot.core.config import load_config
from organizerbot.core.watcher import start_file_watcher
import threading

def main():
    # Load configuration
    config = load_config()
    
    # Start file watcher in a separate thread
    watcher_thread = threading.Thread(
        target=start_file_watcher,
        args=(config.watch_folder,),
        daemon=True
    )
    watcher_thread.start()
    
    # Start the tray application
    run_tray_application()

if __name__ == "__main__":
    main() 