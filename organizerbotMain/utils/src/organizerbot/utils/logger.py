"""
Logging functionality for OrganizerBot
"""
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_action(message: str) -> None:
    """
    Log an action with timestamp
    
    Args:
        message: The message to log
    """
    logging.info(message)
    
    # Also write to a log file
    log_dir = Path.home() / ".organizerbot" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"organizerbot_{datetime.now().strftime('%Y%m%d')}.log"
    with open(log_file, "a") as f:
        f.write(f"{datetime.now().isoformat()} - {message}\n") 