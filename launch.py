"""
OrganizerBot Launcher
"""
import os
import sys

# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), "src")
sys.path.append(src_path)

from organizerbot.core.main import main

if __name__ == "__main__":
    main() 