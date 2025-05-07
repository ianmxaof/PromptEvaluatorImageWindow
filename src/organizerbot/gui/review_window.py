"""
Review window for misclassified images
"""
import customtkinter as ctk
from tkinter import ttk
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk
from typing import Optional, Dict, List
from organizerbot.core.processors import SelfTrainer
from organizerbot.utils.logger import log_action

# Define category groups
CATEGORY_GROUPS = {
    "Body Focused": ["tits", "ass"],
    "Type Based": ["amateur", "professional"],
    "Ethnicity": ["asian", "european", "american"],
    "Special": ["lesbian", "gay", "trans"],
    "Style": ["fetish", "bdsm", "cosplay", "hentai", "manga", "vintage"],
    "General": ["other"]
}

class ReviewWindow:
    def __init__(self, parent: tk.Tk):
        self.window = tk.Toplevel(parent)
        self.window.title("Review Misclassifications")
        self.window.geometry("1000x800")
        self.window.withdraw()  # Hide window initially
        
        self.trainer = SelfTrainer()
        self.current_image: Optional[Path] = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ctk.CTkFrame(self.window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Image display
        self.image_label = ttk.Label(main_frame)
        self.image_label.pack(pady=10)
        
        # Controls
        control_frame = ctk.CTkFrame(main_frame)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(control_frame, text="Previous", command=self.prev_image).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Next", command=self.next_image).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Save", command=self.save_correction).pack(side="left", padx=5)
        
        # Category selection with groups
        self.category_var = tk.StringVar()
        category_frame = ctk.CTkFrame(main_frame)
        category_frame.pack(fill="x", padx=10, pady=10)
        
        # Create notebook for category groups
        notebook = ttk.Notebook(category_frame)
        notebook.pack(fill="x", padx=5, pady=5)
        
        # Create tabs for each group
        for group_name, categories in CATEGORY_GROUPS.items():
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=group_name)
            
            # Add radio buttons for each category in the group
            for category in categories:
                rb = ttk.Radiobutton(
                    tab,
                    text=category.capitalize(),
                    value=category,
                    variable=self.category_var
                )
                rb.pack(anchor="w", padx=20, pady=5)
        
    def show(self):
        """Show the review window"""
        self.window.deiconify()
        self.window.lift()
        
    def hide(self):
        """Hide the review window"""
        self.window.withdraw()
        
    def load_image(self, path: Path):
        """Load and display an image"""
        try:
            image = Image.open(path)
            image.thumbnail((800, 600))  # Increased size for better visibility
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            self.current_image = path
        except Exception as e:
            print(f"Error loading image: {e}")
            
    def prev_image(self):
        """Load previous image"""
        # TODO: Implement previous image loading
        pass
        
    def next_image(self):
        """Load next image"""
        # TODO: Implement next image loading
        pass
        
    def save_correction(self):
        """Save category correction"""
        if self.current_image and self.category_var.get():
            self.trainer.add_training_example(
                self.current_image,
                self.category_var.get()
            )
            self.trainer.train() 