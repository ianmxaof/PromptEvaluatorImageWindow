"""
Launch script for Video Analyzer GUI
"""
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from organizerbot.gui.video_analyzer_gui import VideoAnalyzerGUI
from organizerbot.utils.logger import setup_logging

def main():
    # Setup logging
    setup_logging()
    
    # Launch GUI
    app = VideoAnalyzerGUI()
    app.mainloop()

if __name__ == "__main__":
    main() 