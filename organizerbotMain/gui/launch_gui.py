"""
OrganizerBot GUI Launcher
"""
import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import tkinter
        import PIL
        import ttkthemes
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def main():
    """Main entry point for the GUI launcher"""
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    
    # Add the organizerbotMain directory to Python path
    sys.path.insert(0, str(script_dir))
    
    # Check and install dependencies
    check_dependencies()
    
    # Import and start the GUI
    from organizerbotMain.organizerbot.gui.main_window import start_gui
    start_gui()

if __name__ == "__main__":
    main() 