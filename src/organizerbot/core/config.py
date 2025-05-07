"""
Configuration management for OrganizerBot
"""
import json
from pathlib import Path
from typing import Dict, List
import os

# Define category groups
CATEGORY_GROUPS = {
    "Body Focused": ["tits", "ass"],
    "Type Based": ["amateur", "professional"],
    "Ethnicity": ["asian", "european", "american"],
    "Special": ["lesbian", "gay", "trans"],
    "Style": ["fetish", "bdsm", "cosplay", "hentai", "manga", "vintage"],
    "General": ["other"]
}

class Config:
    def __init__(self):
        self.watch_folder = Path.home() / "Pictures" / "Watch"
        self.source_folder = Path.home() / "Pictures" / "Source"
        self.enable_tray = True
        self.categories = CATEGORY_GROUPS
        
    @classmethod
    def load(cls) -> 'Config':
        """Load configuration from file"""
        config = cls()
        config_file = Path.home() / ".organizerbot" / "config.json"
        
        if config_file.exists():
            try:
                with open(config_file) as f:
                    data = json.load(f)
                    config.watch_folder = Path(data.get("watch_folder", str(config.watch_folder)))
                    config.source_folder = Path(data.get("source_folder", str(config.source_folder)))
                    config.enable_tray = data.get("enable_tray", True)
            except Exception as e:
                print(f"Error loading config: {str(e)}")
                
        return config
        
    def save(self):
        """Save configuration to file"""
        config_file = Path.home() / ".organizerbot" / "config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "watch_folder": str(self.watch_folder),
            "source_folder": str(self.source_folder),
            "enable_tray": self.enable_tray
        }
        
        try:
            with open(config_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {str(e)}")
            
    def ensure_directories(self):
        """Ensure watch and source directories exist"""
        self.watch_folder.mkdir(parents=True, exist_ok=True)
        self.source_folder.mkdir(parents=True, exist_ok=True) 