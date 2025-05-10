"""
Logging configuration for OrganizerBot
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from queue import Queue

# Create a queue for GUI updates
gui_queue = Queue()

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """Setup logging configuration"""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Setup file handler if log file specified
    handlers = [console_handler]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers
    )
    
def log_action(message: str, level: int = logging.INFO):
    """Log an action with the specified level"""
    logger = logging.getLogger("organizerbot")
    logger.log(level, message) 