#!/usr/bin/env python3
"""
Launcher script for video editor GUI
"""
import os
import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.append(str(src_dir))

from organizerbot.gui.video_editor_gui import main

if __name__ == "__main__":
    main() 