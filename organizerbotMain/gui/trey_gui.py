import os
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import pystray

# Initial toggle states
toggles = {
    "watermark_removal": False,
    "enhancement": False,
    "auto_upload": False,
    "watch_folder": os.path.expanduser("~")
}

def toggle_feature(name):
    toggles[name] = not toggles[name]
    print(f"[TOGGLE] {name.replace('_', ' ').title()}: {'ON' if toggles[name] else 'OFF'}")

def choose_watch_folder():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        toggles["watch_folder"] = folder_selected
        print(f"[WATCH FOLDER SET] {folder_selected}")

def create_menu(icon):
    return pystray.Menu(
        pystray.MenuItem(
            lambda item: f"Watermark Removal: {'✅' if toggles['watermark_removal'] else '❌'}",
            lambda: toggle_feature("watermark_removal")),
        pystray.MenuItem(
            lambda item: f"Enhancement: {'✅' if toggles['enhancement'] else '❌'}",
            lambda: toggle_feature("enhancement")),
        pystray.MenuItem(
            lambda item: f"Telegram Auto-Upload: {'✅' if toggles['auto_upload'] else '❌'}",
            lambda: toggle_feature("auto_upload")),
        pystray.MenuItem("Set Watch Folder", choose_watch_folder),
        pystray.MenuItem("Exit", lambda: icon.stop())
    )

def run_tray():
    icon_path = os.path.join(os.path.dirname(__file__), "psicon.png")
    if not os.path.exists(icon_path):
        raise FileNotFoundError("Tray icon 'psicon.png' not found in script directory.")

    image = Image.open(icon_path)
    icon = pystray.Icon("OrganizerBot", image, "OrganizerBot Tray", create_menu(None))
    icon.menu = create_menu(icon)
    icon.run()

if __name__ == "__main__":
    tray_thread = threading.Thread(target=run_tray, daemon=True)
    tray_thread.start()

    print("[OrganizerBot Tray Running] You can minimize this terminal.")
    tray_thread.join()
