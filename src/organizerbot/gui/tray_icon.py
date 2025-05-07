"""
System tray icon for OrganizerBot
"""
import pystray
from PIL import Image, ImageDraw
import tkinter as tk
from typing import Optional
from ..core.processors import ImageProcessor

class TrayIcon:
    def __init__(self, root: tk.Tk, processor: ImageProcessor):
        self.root = root
        self.processor = processor
        self.icon = None
        self.create_icon()
        
    def create_icon(self):
        """Create system tray icon"""
        def create_image(width=64, height=64, color='black', progress=0):
            image = Image.new('RGB', (width, height), color=color)
            draw = ImageDraw.Draw(image)
            
            # Draw progress circle
            angle = int(360 * progress)
            draw.arc([(4, 4), (width-4, height-4)], 0, angle, fill='white', width=4)
            
            return image
            
        def on_click(icon, item):
            if str(item) == "Show":
                self.root.deiconify()
                self.root.lift()
            elif str(item) == "Exit":
                self.destroy()
                self.root.destroy()
                
        menu = pystray.Menu(
            pystray.MenuItem("Show", on_click),
            pystray.MenuItem("Exit", on_click)
        )
        
        self.icon = pystray.Icon(
            "OrganizerBot",
            create_image(),
            "OrganizerBot",
            menu
        )
        
        # Start icon in a separate thread
        import threading
        threading.Thread(target=self.icon.run, daemon=True).start()
        
    def update_progress(self, progress: float):
        """Update progress indicator"""
        if self.icon:
            def create_image(width=64, height=64, color='black'):
                image = Image.new('RGB', (width, height), color=color)
                draw = ImageDraw.Draw(image)
                
                # Draw progress circle
                angle = int(360 * progress)
                draw.arc([(4, 4), (width-4, height-4)], 0, angle, fill='white', width=4)
                
                return image
                
            self.icon.icon = create_image()
            
    def destroy(self):
        """Clean up icon"""
        if self.icon:
            self.icon.stop()
            self.icon = None 