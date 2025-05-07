"""
Utility functions for OrganizerBot
"""
import logging
import sys
from pathlib import Path
import queue

# Global queue for GUI updates
gui_queue = queue.Queue()

def setup_logging(log_file: Path = None, level=logging.INFO):
    """Setup logging configuration"""
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler if log file specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
def log_action(message: str, level=logging.INFO):
    """Log an action and add it to the GUI queue"""
    logging.log(level, message)
    gui_queue.put(message) 