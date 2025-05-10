"""
GUI interface for video analysis
"""
import customtkinter as ctk
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import json
from PIL import Image, ImageTk
import logging
from typing import Optional, Dict, List
import threading
import cv2
import numpy as np
import os
import urllib.parse

from ..processors.video_analyzer import VideoAnalyzer, AnalysisType
from ..utils.logger import setup_logging

logger = logging.getLogger(__name__)

class VideoAnalyzerGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("Video Analyzer")
        self.geometry("1200x800")
        
        # Initialize analyzer
        self.analyzer = VideoAnalyzer()
        self.current_video = None
        self.analysis_thread = None
        self.is_analyzing = False
        
        # Create main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create controls frame
        self.controls_frame = ctk.CTkFrame(self.main_frame)
        self.controls_frame.pack(fill="x", padx=5, pady=5)
        
        # Add file selection button
        self.select_button = ctk.CTkButton(
            self.controls_frame,
            text="Select Video",
            command=self.select_video
        )
        self.select_button.pack(side="left", padx=5)
        
        # Add file path label
        self.file_label = ctk.CTkLabel(
            self.controls_frame,
            text="No video selected",
            wraplength=400
        )
        self.file_label.pack(side="left", padx=5)
        
        # Add analysis type selection
        self.analysis_type = ctk.CTkOptionMenu(
            self.controls_frame,
            values=[t.value for t in AnalysisType],
            command=self.on_analysis_type_change
        )
        self.analysis_type.set(AnalysisType.BASIC.value)
        self.analysis_type.pack(side="left", padx=5)
        
        # Add interval selection
        self.interval_label = ctk.CTkLabel(self.controls_frame, text="Frame Interval (s):")
        self.interval_label.pack(side="left", padx=5)
        self.interval_entry = ctk.CTkEntry(self.controls_frame, width=50)
        self.interval_entry.insert(0, "60")
        self.interval_entry.pack(side="left", padx=5)
        
        # Add analyze button
        self.analyze_button = ctk.CTkButton(
            self.controls_frame,
            text="Analyze",
            command=self.analyze_video,
            state="disabled"
        )
        self.analyze_button.pack(side="left", padx=5)
        
        # Create preview frame
        self.preview_frame = ctk.CTkFrame(self.main_frame)
        self.preview_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add preview label
        self.preview_label = ctk.CTkLabel(
            self.preview_frame,
            text="Select a video file to begin analysis",
            font=("Arial", 14)
        )
        self.preview_label.pack(expand=True)
        
        # Add status label
        self.status_label = ctk.CTkLabel(
            self.main_frame,
            text="Ready",
            font=("Arial", 12)
        )
        self.status_label.pack(side="bottom", pady=5)
        
        # Add progress bar
        self.progress_bar = ctk.CTkProgressBar(self.main_frame)
        self.progress_bar.pack(side="bottom", fill="x", padx=10, pady=5)
        self.progress_bar.set(0)
        
    def select_video(self):
        """Open file dialog to select video"""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mkv *.mov"),
            ("All files", "*.*")
        ]
        
        try:
            # Get initial directory
            initial_dir = None
            if self.current_video:
                initial_dir = str(Path(self.current_video).parent)
            
            filename = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=filetypes,
                initialdir=initial_dir
            )
            
            if filename:
                # Convert to Path object and check if file exists
                video_path = Path(filename).resolve()
                if not video_path.exists():
                    messagebox.showerror("Error", f"File not found: {filename}")
                    return
                    
                # Test if the file can be opened
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        messagebox.showerror("Error", f"Cannot open video file: {filename}")
                        return
                    cap.release()
                except Exception as e:
                    messagebox.showerror("Error", f"Error testing video file: {str(e)}")
                    return
                    
                # Store the path
                self.current_video = str(video_path)
                self.file_label.configure(text=video_path.name)
                self.analyze_button.configure(state="normal")
                self.status_label.configure(text=f"Selected: {video_path.name}")
                
                # Show preview frame
                self.show_preview_frame(video_path)
                
        except Exception as e:
            logger.error(f"Error selecting video: {str(e)}")
            messagebox.showerror("Error", f"Error selecting video: {str(e)}")
            
    def show_preview_frame(self, video_path: Path):
        """Show first frame of the video as preview"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame to fit preview
                preview_width = self.preview_frame.winfo_width() - 20
                preview_height = self.preview_frame.winfo_height() - 20
                
                if preview_width > 0 and preview_height > 0:
                    # Calculate aspect ratio
                    aspect = frame.shape[1] / frame.shape[0]
                    
                    # Calculate new dimensions
                    if preview_width / aspect <= preview_height:
                        new_width = preview_width
                        new_height = int(preview_width / aspect)
                    else:
                        new_height = preview_height
                        new_width = int(preview_height * aspect)
                        
                    # Resize frame
                    frame = cv2.resize(frame, (new_width, new_height))
                    
                # Convert to PIL Image and then to PhotoImage
                image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image)
                
                # Update preview
                self.preview_label.configure(image=photo, text="")
                self.preview_label.image = photo
            else:
                self.preview_label.configure(text="Could not load video preview")
                
        except Exception as e:
            logger.error(f"Error showing preview: {str(e)}")
            self.preview_label.configure(text="Error loading preview")
            
    def on_analysis_type_change(self, choice):
        """Handle analysis type change"""
        pass
        
    def analyze_video(self):
        """Start video analysis"""
        if not self.current_video:
            messagebox.showerror("Error", "Please select a video file first")
            return
            
        if self.is_analyzing:
            return
            
        try:
            # Get analysis parameters
            interval = int(self.interval_entry.get())
            analysis_type = AnalysisType(self.analysis_type.get())
            
            # Disable controls during analysis
            self.analyze_button.configure(state="disabled")
            self.select_button.configure(state="disabled")
            self.is_analyzing = True
            self.progress_bar.set(0)
            self.status_label.configure(text="Analyzing video...")
            
            # Start analysis in background thread
            self.analysis_thread = threading.Thread(
                target=self.run_analysis,
                args=(interval, analysis_type)
            )
            self.analysis_thread.start()
            
            # Start progress update
            self.after(100, self.update_progress)
            
        except ValueError:
            messagebox.showerror("Error", "Invalid interval value")
            self.analyze_button.configure(state="normal")
            self.select_button.configure(state="normal")
            self.is_analyzing = False
            
    def run_analysis(self, interval: int, analysis_type: AnalysisType):
        """Run video analysis in background thread"""
        try:
            # Run analysis
            manifest_path, grid_path = self.analyzer.analyze_video(
                self.current_video,
                interval=interval,
                show_grid=True,
                analysis_type=analysis_type
            )
            
            # Update GUI in main thread
            self.after(0, self.analysis_complete, manifest_path, grid_path)
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            self.after(0, self.analysis_failed, str(e))
            
    def analysis_complete(self, manifest_path: str, grid_path: str = None):
        """Handle analysis completion"""
        try:
            # Show results
            if grid_path and os.path.exists(grid_path):
                # Load and display grid image
                grid_img = Image.open(grid_path)
                # Resize to fit preview frame
                preview_width = self.preview_frame.winfo_width() - 20
                preview_height = self.preview_frame.winfo_height() - 20
                
                if preview_width > 0 and preview_height > 0:
                    # Calculate aspect ratio
                    aspect = grid_img.size[0] / grid_img.size[1]
                    
                    # Calculate new dimensions
                    if preview_width / aspect <= preview_height:
                        new_width = preview_width
                        new_height = int(preview_width / aspect)
                    else:
                        new_height = preview_height
                        new_width = int(preview_height * aspect)
                        
                    # Resize image
                    grid_img = grid_img.resize((new_width, new_height))
                    
                photo = ImageTk.PhotoImage(grid_img)
                self.preview_label.configure(image=photo)
                self.preview_label.image = photo  # Keep reference
            else:
                self.preview_label.configure(
                    text="Analysis complete. No preview available."
                )
                
            # Show success message
            messagebox.showinfo(
                "Success",
                f"Analysis complete!\nResults saved to: {manifest_path}"
            )
            
        finally:
            # Re-enable controls
            self.analyze_button.configure(state="normal")
            self.select_button.configure(state="normal")
            self.is_analyzing = False
            self.progress_bar.set(1)
            self.status_label.configure(text="Analysis complete")
            
    def analysis_failed(self, error_msg: str):
        """Handle analysis failure"""
        messagebox.showerror("Error", f"Analysis failed: {error_msg}")
        self.analyze_button.configure(state="normal")
        self.select_button.configure(state="normal")
        self.is_analyzing = False
        self.progress_bar.set(0)
        self.status_label.configure(text="Analysis failed")
        
    def update_progress(self):
        """Update progress bar animation"""
        if self.is_analyzing:
            current = self.progress_bar.get()
            if current >= 1:
                current = 0
            self.progress_bar.set(current + 0.01)
            self.after(50, self.update_progress)
            
def main():
    setup_logging()
    app = VideoAnalyzerGUI()
    app.mainloop()
    
if __name__ == "__main__":
    main() 