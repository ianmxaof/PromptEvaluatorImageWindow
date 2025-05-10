"""
Standalone GUI window for PowerCoreAi
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
from organizerbot.core.config import load_config, save_config
from organizerbot.utils.logger import log_action

class MainWindow:
    """Main application window"""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PowerCoreAi")
        self.root.geometry("800x600")
        self.config = load_config()
        
        self.setup_ui()
        self.load_config()

    def setup_ui(self):
        """Setup the user interface"""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Source Folders Section
        folders_frame = ttk.LabelFrame(main_frame, text="Source Folders", padding="5")
        folders_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.folders_list = tk.Listbox(folders_frame, height=5)
        self.folders_list.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        buttons_frame = ttk.Frame(folders_frame)
        buttons_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(buttons_frame, text="Add Folder", command=self.add_folder).grid(row=0, column=0, padx=5)
        ttk.Button(buttons_frame, text="Remove Folder", command=self.remove_folder).grid(row=0, column=1, padx=5)
        
        # Features Section
        features_frame = ttk.LabelFrame(main_frame, text="Features", padding="5")
        features_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.watermark_var = tk.BooleanVar()
        self.enhancement_var = tk.BooleanVar()
        self.auto_upload_var = tk.BooleanVar()
        
        ttk.Checkbutton(features_frame, text="Watermark Removal", variable=self.watermark_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(features_frame, text="Image Enhancement", variable=self.enhancement_var).grid(row=1, column=0, sticky=tk.W)
        ttk.Checkbutton(features_frame, text="Auto Upload to Telegram", variable=self.auto_upload_var).grid(row=2, column=0, sticky=tk.W)
        
        # Categories Section
        categories_frame = ttk.LabelFrame(main_frame, text="Categories", padding="5")
        categories_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.categories_list = tk.Listbox(categories_frame, height=5)
        self.categories_list.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        cat_buttons_frame = ttk.Frame(categories_frame)
        cat_buttons_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(cat_buttons_frame, text="Add Category", command=self.add_category).grid(row=0, column=0, padx=5)
        ttk.Button(cat_buttons_frame, text="Remove Category", command=self.remove_category).grid(row=0, column=1, padx=5)
        
        # Save Button
        ttk.Button(main_frame, text="Save Settings", command=self.save_settings).grid(row=3, column=0, pady=10)

    def load_config(self):
        """Load configuration into UI"""
        # Load folders
        self.folders_list.delete(0, tk.END)
        for folder in self.config.get('watch_folders', []):
            self.folders_list.insert(tk.END, folder)
        
        # Load features
        self.watermark_var.set(self.config['features'].get('watermark_removal', False))
        self.enhancement_var.set(self.config['features'].get('enhancement', False))
        self.auto_upload_var.set(self.config['features'].get('auto_upload', False))
        
        # Load categories
        self.categories_list.delete(0, tk.END)
        for category in self.config.get('categories', []):
            self.categories_list.insert(tk.END, category)

    def add_folder(self):
        """Add a new source folder"""
        folder = filedialog.askdirectory()
        if folder:
            self.folders_list.insert(tk.END, folder)

    def remove_folder(self):
        """Remove selected folder"""
        selection = self.folders_list.curselection()
        if selection:
            self.folders_list.delete(selection[0])

    def add_category(self):
        """Add a new category"""
        category = tk.simpledialog.askstring("Add Category", "Enter category name:")
        if category:
            self.categories_list.insert(tk.END, category)

    def remove_category(self):
        """Remove selected category"""
        selection = self.categories_list.curselection()
        if selection:
            self.categories_list.delete(selection[0])

    def save_settings(self):
        """Save current settings"""
        # Update folders
        self.config['watch_folders'] = list(self.folders_list.get(0, tk.END))
        
        # Update features
        self.config['features'] = {
            'watermark_removal': self.watermark_var.get(),
            'enhancement': self.enhancement_var.get(),
            'auto_upload': self.auto_upload_var.get()
        }
        
        # Update categories
        self.config['categories'] = list(self.categories_list.get(0, tk.END))
        
        # Save to file
        save_config(self.config)
        messagebox.showinfo("Success", "Settings saved successfully!")
        log_action("Settings saved from GUI")

    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

def start_gui():
    """Start the GUI application"""
    app = MainWindow()
    app.run() 