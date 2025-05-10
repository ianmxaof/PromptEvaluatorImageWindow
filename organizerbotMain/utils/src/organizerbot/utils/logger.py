"""
Logging functionality for PowerCoreAi
"""
import logging
import os
from pathlib import Path
from datetime import datetime

# Create logs directory in user's home
log_dir = Path.home() / ".powercoreai" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# Configure logging
log_file = log_dir / f"powercoreai_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("powercoreai")

def log_action(message: str) -> None:
    """
    Log an action with timestamp
    
    Args:
        message: The message to log
    """
    logging.info(message)
    
    # Also write to a log file
    log_dir = Path.home() / ".powercoreai" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"powercoreai_{datetime.now().strftime('%Y%m%d')}.log"
    with open(log_file, "a") as f:
        f.write(f"{datetime.now().isoformat()} - {message}\n") 