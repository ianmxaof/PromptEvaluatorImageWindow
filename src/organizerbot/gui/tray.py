"""
System tray interface for OrganizerBot
"""
import os
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw
import pystray
from organizerbot.core.config import load_config, save_config

class TrayApplication:
    """System tray application"""
    def __init__(self):
        self.config = load_config()
        self.icon = None

    def create_default_icon(self):
        """Create a default icon if none exists"""
        # Create a 64x64 image with a dark background
        image = Image.new('RGB', (64, 64), color='black')
        draw = ImageDraw.Draw(image)
        
        # Draw a simple "O" in white
        draw.ellipse([16, 16, 48, 48], outline='white', width=2)
        
        return image

    def toggle_feature(self, name: str) -> None:
        """Toggle a feature on/off"""
        self.config.features[name] = not self.config.features[name]
        save_config(self.config)
        print(f"[TOGGLE] {name.replace('_', ' ').title()}: {'ON' if self.config.features[name] else 'OFF'}")

    def choose_watch_folder(self) -> None:
        """Open folder selection dialog"""
        root = tk.Tk()
        root.withdraw()
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.config.watch_folder = folder_selected
            save_config(self.config)
            print(f"[WATCH FOLDER SET] {folder_selected}")

    def list_sources(self) -> None:
        """List all configured source folders"""
        print("[SOURCE FOLDERS]")
        for i, source in enumerate(self.config.source_folders, start=1):
            print(f"{i}. {source}")

    def create_menu(self) -> pystray.Menu:
        """Create the system tray menu"""
        return pystray.Menu(
            pystray.MenuItem(
                lambda item: f"Watermark Removal: {'✅' if self.config.features['watermark_removal'] else '❌'}",
                lambda: self.toggle_feature("watermark_removal")),
            pystray.MenuItem(
                lambda item: f"Enhancement: {'✅' if self.config.features['enhancement'] else '❌'}",
                lambda: self.toggle_feature("enhancement")),
            pystray.MenuItem(
                lambda item: f"Auto-Upload: {'✅' if self.config.features['auto_upload'] else '❌'}",
                lambda: self.toggle_feature("auto_upload")),
            pystray.MenuItem("Set Watch Folder", self.choose_watch_folder),
            pystray.MenuItem("List Source Folders", self.list_sources),
            pystray.MenuItem("Exit", lambda: self.icon.stop())
        )

    def run(self) -> None:
        """Start the tray application"""
        image = self.create_default_icon()
        self.icon = pystray.Icon("OrganizerBot", image, "OrganizerBot Tray", self.create_menu())
        self.icon.run()

def run_tray_application() -> None:
    """Start the tray application in a separate thread"""
    tray_app = TrayApplication()
    tray_thread = threading.Thread(target=tray_app.run, daemon=True)
    tray_thread.start()
    print("[OrganizerBot Tray Running] You can minimize this terminal.")
    tray_thread.join() 