"""
Main GUI window for OrganizerBot
"""
import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
from ..core import Config
from ..core import FileEventHandler, start_file_watcher
from ..utils.logger import log_action, gui_queue
import threading
import queue
import time
from collections import defaultdict
from pathlib import Path
from .review_window import ReviewWindow
import pystray
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import ttk
import logging
from typing import Dict, Optional
from ..core import ImageProcessor, SelfTrainer
from ..utils import setup_logging
from .tray_icon import TrayIcon

# Define category groups
CATEGORY_GROUPS = {
    "Body Focused": ["tits", "ass"],
    "Type Based": ["amateur", "professional"],
    "Ethnicity": ["asian", "european", "american"],
    "Special": ["lesbian", "gay", "trans"],
    "Style": ["fetish", "bdsm", "cosplay", "hentai", "manga", "vintage"],
    "General": ["other"]
}

class MainWindow:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = Config.load()
        self.processor = ImageProcessor()
        self.self_trainer = SelfTrainer()
        self.tray_icon = None
        self.is_watching = False
        self.category_counts = {}
        self.last_counts = {}
        self.activity_log = []
        self.last_processed_file = None
        
        # Initialize main window
        self.root = ctk.CTk()
        self.root.title("OrganizerBot")
        self.root.geometry("1000x800")  # Increased size for more categories
        
        # Configure grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=0)  # Header
        self.root.grid_rowconfigure(1, weight=1)  # Main content
        self.root.grid_rowconfigure(2, weight=0)  # Activity panel
        self.root.grid_rowconfigure(3, weight=0)  # Footer
        
        # Create header with centered title and status
        self.header_frame = ctk.CTkFrame(self.root)
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=10)
        self.header_frame.grid_columnconfigure(0, weight=1)
        
        # Title and status container
        self.title_status_frame = ctk.CTkFrame(self.header_frame)
        self.title_status_frame.grid(row=0, column=0, padx=20, pady=10)
        self.title_status_frame.grid_columnconfigure(0, weight=1)
        self.title_status_frame.grid_columnconfigure(1, weight=1)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.title_status_frame,
            text="Category Statistics",
            font=("Helvetica", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, padx=(0, 10))
        
        # Status bar (reduced width)
        self.status_bar = ctk.CTkProgressBar(self.title_status_frame, width=200)
        self.status_bar.grid(row=0, column=1, padx=(10, 0))
        self.status_bar.set(0)
        
        # Create main content area
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        
        # Left column - Settings
        self.settings_frame = ctk.CTkFrame(self.main_frame)
        self.settings_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=10)
        
        # Watch folder selection
        self.watch_folder_frame = ctk.CTkFrame(self.settings_frame)
        self.watch_folder_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        self.watch_folder_frame.grid_columnconfigure(0, weight=1)
        
        self.watch_folder_label = ctk.CTkLabel(self.watch_folder_frame, text="Watch Folder:")
        self.watch_folder_label.grid(row=0, column=0, sticky="w")
        
        self.watch_folder_entry = ctk.CTkEntry(self.watch_folder_frame)
        self.watch_folder_entry.grid(row=1, column=0, sticky="ew", padx=(0, 5))
        self.watch_folder_entry.insert(0, str(self.config.watch_folder))
        
        self.watch_folder_button = ctk.CTkButton(
            self.watch_folder_frame,
            text="Browse",
            width=80,
            command=self.select_watch_folder
        )
        self.watch_folder_button.grid(row=1, column=1, padx=(5, 0))
        
        # Source folder selection
        self.source_folder_frame = ctk.CTkFrame(self.settings_frame)
        self.source_folder_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        self.source_folder_frame.grid_columnconfigure(0, weight=1)
        
        self.source_folder_label = ctk.CTkLabel(self.source_folder_frame, text="Source Folder:")
        self.source_folder_label.grid(row=0, column=0, sticky="w")
        
        self.source_folder_entry = ctk.CTkEntry(self.source_folder_frame)
        self.source_folder_entry.grid(row=1, column=0, sticky="ew", padx=(0, 5))
        self.source_folder_entry.insert(0, str(self.config.source_folder))
        
        self.source_folder_button = ctk.CTkButton(
            self.source_folder_frame,
            text="Browse",
            width=80,
            command=self.select_source_folder
        )
        self.source_folder_button.grid(row=1, column=1, padx=(5, 0))
        
        # Control buttons with fixed width
        self.control_frame = ctk.CTkFrame(self.settings_frame)
        self.control_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        self.control_frame.grid_columnconfigure(0, weight=1)
        self.control_frame.grid_columnconfigure(1, weight=1)
        
        self.start_button = ctk.CTkButton(
            self.control_frame,
            text="Start Watching",
            width=120,
            command=self.toggle_watching
        )
        self.start_button.grid(row=0, column=0, padx=5, pady=5)
        
        self.refresh_button = ctk.CTkButton(
            self.control_frame,
            text="Manual Refresh",
            width=120,
            command=self.manual_refresh
        )
        self.refresh_button.grid(row=0, column=1, padx=5, pady=5)
        
        # Feedback button
        self.feedback_button = ctk.CTkButton(
            self.control_frame,
            text="Provide Feedback",
            width=120,
            command=self.show_feedback_dialog
        )
        self.feedback_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        
        # Training stats
        self.stats_frame = ctk.CTkFrame(self.settings_frame)
        self.stats_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        
        self.stats_label = ctk.CTkLabel(
            self.stats_frame,
            text="Training Statistics",
            font=("Helvetica", 14, "bold")
        )
        self.stats_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        
        self.stats_text = ctk.CTkTextbox(self.stats_frame, height=100)
        self.stats_text.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        self.stats_text.grid_columnconfigure(0, weight=1)
        
        # Tray icon toggle with tooltip
        self.tray_frame = ctk.CTkFrame(self.settings_frame)
        self.tray_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=5)
        
        self.tray_switch = ctk.CTkSwitch(
            self.tray_frame,
            text="Enable Tray",
            command=self.toggle_tray
        )
        self.tray_switch.grid(row=0, column=0, sticky="w")
        
        # Create tooltip
        self.tooltip = None
        self.tray_switch.bind("<Enter>", self.show_tooltip)
        self.tray_switch.bind("<Leave>", self.hide_tooltip)
        
        # Right column - Category statistics with scrollable frame
        self.stats_frame = ctk.CTkScrollableFrame(self.main_frame)
        self.stats_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=10)
        
        # Create category labels with right-aligned counts
        self.category_labels = {}
        row = 0
        
        for group_name, categories in self.config.categories.items():
            # Group header
            group_frame = ctk.CTkFrame(self.stats_frame)
            group_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=(10, 5))
            group_frame.grid_columnconfigure(0, weight=1)
            
            group_label = ctk.CTkLabel(
                group_frame,
                text=group_name,
                font=("Helvetica", 14, "bold")
            )
            group_label.grid(row=0, column=0, sticky="w")
            row += 1
            
            # Categories in group
            for category in categories:
                frame = ctk.CTkFrame(self.stats_frame)
                frame.grid(row=row, column=0, sticky="ew", padx=10, pady=2)
                frame.grid_columnconfigure(0, weight=1)
                frame.grid_columnconfigure(1, weight=0)
                
                # Category name
                label = ctk.CTkLabel(frame, text=category.capitalize())
                label.grid(row=0, column=0, sticky="w")
                
                # Count with right alignment
                count_label = ctk.CTkLabel(frame, text="0", width=50)
                count_label.grid(row=0, column=1, sticky="e", padx=(10, 0))
                
                self.category_labels[category] = count_label
                self.category_counts[category] = 0
                self.last_counts[category] = 0
                row += 1
        
        # Create activity panel
        self.activity_frame = ctk.CTkFrame(self.root)
        self.activity_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=10)
        self.activity_frame.grid_columnconfigure(0, weight=1)
        
        # Activity panel header
        self.activity_header = ctk.CTkLabel(
            self.activity_frame,
            text="Recent Activity",
            font=("Helvetica", 14, "bold")
        )
        self.activity_header.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))
        
        # Activity log with scrollbar
        self.activity_log_frame = ctk.CTkScrollableFrame(self.activity_frame, height=150)
        self.activity_log_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.activity_log_frame.grid_columnconfigure(0, weight=1)
        
        # Create footer
        self.footer_frame = ctk.CTkFrame(self.root)
        self.footer_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=10)
        
        # Status label
        self.status_label = ctk.CTkLabel(self.footer_frame, text="Ready")
        self.status_label.grid(row=0, column=0, sticky="w")
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start with tray icon if enabled
        if self.config.enable_tray:
            self.tray_switch.select()
            self.toggle_tray()
            
        # Start activity log update thread
        self.start_activity_log_updater()
        
        # Update training stats periodically
        self.update_training_stats()

    def add_activity_log(self, message: str):
        """Add a message to the activity log"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.activity_log.append(log_entry)
        
        # Keep only last 100 entries
        if len(self.activity_log) > 100:
            self.activity_log = self.activity_log[-100:]
            
        # Update display
        self.update_activity_display()
        
    def update_activity_display(self):
        """Update the activity log display"""
        # Clear existing widgets
        for widget in self.activity_log_frame.winfo_children():
            widget.destroy()
            
        # Add log entries
        for i, entry in enumerate(reversed(self.activity_log)):
            label = ctk.CTkLabel(
                self.activity_log_frame,
                text=entry,
                anchor="w",
                justify="left"
            )
            label.grid(row=i, column=0, sticky="ew", padx=5, pady=2)
            
    def start_activity_log_updater(self):
        """Start thread to update activity log from queue"""
        def update_from_queue():
            while True:
                try:
                    message = gui_queue.get_nowait()
                    self.add_activity_log(message)
                except queue.Empty:
                    break
                time.sleep(0.1)
            self.root.after(100, update_from_queue)
            
        self.root.after(100, update_from_queue)

    def show_tooltip(self, event):
        if self.tooltip is None:
            x, y, _, _ = self.tray_switch.bbox("insert")
            x += self.tray_switch.winfo_rootx() + 25
            y += self.tray_switch.winfo_rooty() - 25
            
            self.tooltip = tk.Toplevel(self.root)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            
            label = tk.Label(
                self.tooltip,
                text="Show system tray icon with progress indicator",
                justify=tk.LEFT,
                background="#ffffe0",
                relief=tk.SOLID,
                borderwidth=1,
                font=("Helvetica", "10")
            )
            label.pack()

    def hide_tooltip(self, event):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

    def select_watch_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.config.watch_folder = Path(folder)
            self.watch_folder_entry.delete(0, tk.END)
            self.watch_folder_entry.insert(0, str(folder))
            self.config.save()
            self.add_activity_log(f"Watch folder set to: {folder}")

    def select_source_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.config.source_folder = Path(folder)
            self.source_folder_entry.delete(0, tk.END)
            self.source_folder_entry.insert(0, str(folder))
            self.config.save()
            self.add_activity_log(f"Source folder set to: {folder}")

    def toggle_watching(self):
        if not self.is_watching:
            self.start_watching()
        else:
            self.stop_watching()

    def start_watching(self):
        if not self.config.watch_folder.exists():
            messagebox.showerror("Error", "Watch folder does not exist")
            return
            
        if not self.config.source_folder.exists():
            messagebox.showerror("Error", "Source folder does not exist")
            return
            
        self.is_watching = True
        self.start_button.configure(text="Stop Watching")
        self.status_label.configure(text="Watching for new files...")
        self.add_activity_log("Started watching for new files")
        
        # Start file watcher
        self.observer = start_file_watcher(
            self.config.watch_folder,
            self.process_new_file
        )
        
    def stop_watching(self):
        if hasattr(self, 'observer'):
            self.observer.stop()
            self.observer.join()
            
        self.is_watching = False
        self.start_button.configure(text="Start Watching")
        self.status_label.configure(text="Ready")
        self.add_activity_log("Stopped watching for new files")

    def process_new_file(self, file_path: str):
        try:
            # Process the image
            category, metadata = self.processor.process_image(file_path)
            
            # Update counts
            self.category_counts[category] = self.category_counts.get(category, 0) + 1
            self.update_category_display(category)
            
            # Store last processed file for feedback
            self.last_processed_file = file_path
            
            # Log action
            log_action(f"Processed {file_path} as {category}")
            self.add_activity_log(f"Processed {Path(file_path).name} as {category}")
            
        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            log_action(error_msg)
            self.add_activity_log(error_msg)

    def manual_refresh(self):
        if not self.is_watching:
            messagebox.showerror("Error", "Start watching first")
            return
            
        self.status_label.configure(text="Refreshing...")
        self.add_activity_log("Starting manual refresh")
        
        # Process all files in watch folder
        for file_path in self.config.watch_folder.glob("**/*"):
            if file_path.is_file():
                self.process_new_file(str(file_path))
                
        self.status_label.configure(text="Refresh complete")
        self.add_activity_log("Manual refresh completed")

    def update_status(self, progress: float, category: Optional[str] = None):
        self.status_bar.set(progress)
        if category:
            self.status_label.configure(text=f"Processing {category}...")

    def update_category_display(self, category: str):
        if category in self.category_labels:
            count = self.category_counts.get(category, 0)
            self.category_labels[category].configure(text=str(count))
            
            # Show green increment if count increased
            if count > self.last_counts[category]:
                self.category_labels[category].configure(text_color="green")
                self.root.after(1000, lambda: self.category_labels[category].configure(text_color="white"))
                self.last_counts[category] = count

    def toggle_tray(self):
        if self.tray_switch.get():
            if not self.tray_icon:
                self.tray_icon = TrayIcon(self.root, self.processor)
            self.config.enable_tray = True
            self.add_activity_log("Tray icon enabled")
        else:
            if self.tray_icon:
                self.tray_icon.destroy()
                self.tray_icon = None
            self.config.enable_tray = False
            self.add_activity_log("Tray icon disabled")
        self.config.save()

    def on_closing(self):
        if self.is_watching:
            self.stop_watching()
        if self.tray_icon:
            self.tray_icon.destroy()
        self.root.destroy()

    def show_feedback_dialog(self):
        """Show dialog for providing feedback on last processed file"""
        if not self.last_processed_file:
            messagebox.showinfo("Feedback", "No file has been processed yet")
            return
            
        # Create feedback dialog
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Provide Feedback")
        dialog.geometry("600x800")  # Increased size for better preview
        dialog.transient(self.root)  # Make dialog stay on top
        dialog.grab_set()  # Make dialog modal
        
        # File info with preview
        file_frame = ctk.CTkFrame(dialog)
        file_frame.pack(fill="x", padx=20, pady=10)
        
        file_label = ctk.CTkLabel(
            file_frame,
            text=f"File: {Path(self.last_processed_file).name}",
            wraplength=550,
            font=("Helvetica", 12, "bold")
        )
        file_label.pack(padx=10, pady=5)
        
        # Try to show image preview
        try:
            # Create preview frame
            preview_frame = ctk.CTkFrame(file_frame)
            preview_frame.pack(padx=10, pady=5, fill="both", expand=True)
            
            # Load and resize image
            preview = Image.open(self.last_processed_file)
            
            # Calculate aspect ratio preserving size
            max_size = (400, 400)
            preview.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ctk.CTkImage(
                light_image=preview,
                dark_image=preview,
                size=preview.size
            )
            
            # Create label with image
            preview_label = ctk.CTkLabel(
                preview_frame,
                image=photo,
                text="",
                width=preview.size[0],
                height=preview.size[1]
            )
            preview_label.pack(padx=10, pady=5)
            
            # Add border
            preview_frame.configure(border_width=2, border_color="gray")
            
        except Exception as e:
            self.logger.error(f"Error showing preview: {str(e)}")
            error_label = ctk.CTkLabel(
                file_frame,
                text=f"Could not load image preview: {str(e)}",
                text_color="red"
            )
            error_label.pack(padx=10, pady=5)
        
        # Category selection
        category_frame = ctk.CTkFrame(dialog)
        category_frame.pack(fill="x", padx=20, pady=10)
        
        category_label = ctk.CTkLabel(
            category_frame,
            text="Select Correct Category:",
            font=("Helvetica", 12, "bold")
        )
        category_label.pack(padx=10, pady=(10, 5))
        
        # Create scrollable frame for categories
        scroll_frame = ctk.CTkScrollableFrame(category_frame, height=300)
        scroll_frame.pack(fill="x", padx=10, pady=5)
        
        # Create category selection with groups
        category_var = ctk.StringVar()
        
        for group_name, categories in CATEGORY_GROUPS.items():
            # Group label
            group_label = ctk.CTkLabel(
                scroll_frame,
                text=group_name,
                font=("Helvetica", 11, "bold")
            )
            group_label.pack(padx=10, pady=(10, 5), anchor="w")
            
            # Categories in group
            for category in categories:
                radio = ctk.CTkRadioButton(
                    scroll_frame,
                    text=category.capitalize(),
                    variable=category_var,
                    value=category
                )
                radio.pack(padx=20, pady=2, anchor="w")
        
        # Submit button
        def submit_feedback():
            category = category_var.get()
            if not category:
                messagebox.showwarning("Warning", "Please select a category")
                return
                
            self.processor.provide_feedback(self.last_processed_file, category)
            self.add_activity_log(f"Provided feedback for {Path(self.last_processed_file).name}: {category}")
            self.update_training_stats()
            dialog.destroy()
            
        button_frame = ctk.CTkFrame(dialog)
        button_frame.pack(fill="x", padx=20, pady=20)
        
        submit_button = ctk.CTkButton(
            button_frame,
            text="Submit Feedback",
            command=submit_feedback,
            width=200,
            height=40,
            font=("Helvetica", 12, "bold")
        )
        submit_button.pack(pady=10)
        
        # Center dialog on screen
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'{width}x{height}+{x}+{y}')

    def update_training_stats(self):
        """Update training statistics display"""
        try:
            stats = self.processor.get_training_stats()
            
            # Format stats text
            text = f"Total Examples: {stats['total_examples']}\n"
            text += "Categories:\n"
            for cat, count in stats['categories'].items():
                text += f"  {cat}: {count}\n"
            if stats['latest_model']:
                text += f"\nLatest Model: {Path(stats['latest_model']).name}"
                
            self.stats_text.delete("1.0", tk.END)
            self.stats_text.insert("1.0", text)
            
        except Exception as e:
            self.logger.error(f"Error updating training stats: {str(e)}")
            
        # Schedule next update
        self.root.after(5000, self.update_training_stats)

    def run(self):
        self.root.mainloop()

def run_gui():
    """Start the GUI"""
    window = MainWindow()
    window.run() 