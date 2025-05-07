# OrganizerBot

A Python-based desktop application for automated file organization and processing.

## Features
- System tray interface for easy access
- Standalone GUI for configuration
- Automatic file monitoring and processing
- AI-powered image categorization
- Image enhancement capabilities
- Watermark removal
- Telegram integration
- Customizable categories

## Project Structure
```
OrganizerBot-master/
├── organizerbotMain/          # Main application code
│   ├── organizerbot/
│   │   ├── core/             # Core application logic
│   │   ├── gui/              # User interface components
│   │   ├── processors/       # File processing modules
│   │   ├── services/         # External service integrations
│   │   └── utils/            # Utility functions
│   └── main.py              # Application entry point
├── deprecated/               # Old/unused files
├── launch_gui.py            # GUI launcher script
├── launch_gui.bat           # Windows launcher
├── launch_gui.sh            # Linux/macOS launcher
└── requirements.txt         # Python dependencies
```

## Setup
1. Install Python 3.x
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   - Windows: Double-click `launch_gui.bat`
   - Linux/macOS: Run `./launch_gui.sh`
   - Or run directly: `python launch_gui.py`

## Configuration
- Configure through the standalone GUI
- Set source folders for image organization
- Customize categories
- Toggle features on/off
- Set up Telegram integration

## Telegram Integration
- Interactive command system
- Photo processing and categorization
- Folder management
- Real-time status updates

## AI Features
- CLIP-based image categorization
- Customizable categories
- Confidence scoring
- Top-k category suggestions

## License
MIT License 