"""
GUI interface for video editing toolkit
"""
import customtkinter as ctk
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import json
from PIL import Image, ImageTk, ImageEnhance, ImageDraw
import logging
from typing import Optional, Dict, List, Tuple
import threading
import cv2
import numpy as np
import os
import urllib.parse
import time
from datetime import timedelta

from ..processors.video_editor import VideoEditor, EditAction
from ..models.data_models import (
    VideoMetadata,
    ProcessingConfig,
    VideoSegment,
    ProcessingResult,
    UserConfig,
    ProcessingStats,
    SecurityConfig
)
from ..utils.logger import setup_logging

logger = logging.getLogger(__name__)

class VideoEditorGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Initialize tooltip tracking
        self.active_tooltip = None
        
        # Configure window
        self.title("Video Editor")
        self.geometry("550x600")
        self.minsize(550, 600)  # Set minimum window size
        
        # Initialize editor with config
        self.config = ProcessingConfig(
            output_dir=Path.home() / ".organizerbot" / "output",
            frames_dir=Path.home() / ".organizerbot" / "output" / "frames"
        )
        self.editor = VideoEditor(config=self.config)
        self.current_video = None
        self.current_segments = []
        self.current_segment = None
        self.is_editing = False
        self.analysis_progress = 0
        self.is_playing = False
        self.current_time = 0
        self.video_cap = None
        self.fps = 0
        self.total_frames = 0
        self.duration = 0
        
        # Create main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create top controls container
        self.top_controls = ctk.CTkFrame(
            self.main_frame,
            fg_color="transparent",
            corner_radius=10,
            border_width=1,
            border_color="gray40"
        )
        self.top_controls.pack(fill="x", padx=10, pady=(5, 0))
        
        # Add file selection and analysis controls (left side)
        self.left_controls = ctk.CTkFrame(
            self.top_controls,
            fg_color="transparent",
            corner_radius=10,
            border_width=1,
            border_color="gray40"
        )
        self.left_controls.pack(side="left", padx=5, pady=5)
        
        # Add file selection button
        self.select_button = ctk.CTkButton(
            self.left_controls,
            text="Select Video",
            command=self.select_video,
            width=75,
            height=32
        )
        self.select_button.pack(pady=(10, 5))
        
        # Add analysis dropdown
        self.analysis_mode = ctk.StringVar(value="Select Analysis Mode")
        self.analysis_dropdown = ctk.CTkOptionMenu(
            self.left_controls,
            values=[
                "Basic Analysis",
                "Scene Analysis",
                "Motion Analysis",
                "Full Analysis"
            ],
            variable=self.analysis_mode,
            command=self.start_analysis,
            width=75,
            state="disabled"
        )
        self.analysis_dropdown.pack(pady=(0, 10))
        
        # Add file path label
        self.file_label = ctk.CTkLabel(
            self.left_controls,
            text="No video selected",
            wraplength=75,  # Reduced to match container width
            font=("Arial", 11)
        )
        self.file_label.pack(pady=(0, 10))
        
        # Create edit buttons container (right side)
        self.edit_container = ctk.CTkFrame(
            self.top_controls,
            fg_color="transparent",
            corner_radius=10,
            border_width=1,
            border_color="gray40"
        )
        self.edit_container.pack(side="right", padx=5, pady=5)
        
        # Create edit buttons grid
        self.edit_grid = ctk.CTkFrame(
            self.edit_container,
            fg_color="transparent"
        )
        self.edit_grid.pack(padx=10, pady=10)
        
        # Add edit buttons in 2x2 grid with normalized size
        button_size = 32
        
        self.split_button = ctk.CTkButton(
            self.edit_grid,
            text="✂",  # Scissors symbol
            command=lambda: self.start_edit(EditAction.SPLIT),
            state="disabled",
            width=button_size,
            height=button_size,
            font=("Arial", 16)
        )
        self.split_button.grid(row=0, column=0, padx=2, pady=2)
        self.create_tooltip(self.split_button, "Split at Current Position")
        
        self.crop_button = ctk.CTkButton(
            self.edit_grid,
            text="⬛",  # Square symbol
            command=lambda: self.start_edit(EditAction.CROP),
            state="disabled",
            width=button_size,
            height=button_size,
            font=("Arial", 16)
        )
        self.crop_button.grid(row=0, column=1, padx=2, pady=2)
        self.create_tooltip(self.crop_button, "Crop Video")
        
        self.delete_button = ctk.CTkButton(
            self.edit_grid,
            text="❌",  # X symbol
            command=lambda: self.start_edit(EditAction.DELETE),
            state="disabled",
            width=button_size,
            height=button_size,
            font=("Arial", 16)
        )
        self.delete_button.grid(row=1, column=0, padx=2, pady=2)
        self.create_tooltip(self.delete_button, "Delete Segment")
        
        self.save_button = ctk.CTkButton(
            self.edit_grid,
            text="💾",  # Floppy disk symbol
            command=lambda: self.start_edit(EditAction.SAVE),
            state="disabled",
            width=button_size,
            height=button_size,
            font=("Arial", 16)
        )
        self.save_button.grid(row=1, column=1, padx=2, pady=2)
        self.create_tooltip(self.save_button, "Save Changes")
        
        # Add scrub slider and time display
        self.scrub_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color="transparent"
        )
        self.scrub_frame.pack(fill="x", padx=10, pady=2)
        
        # Add time display
        self.time_label = ctk.CTkLabel(
            self.scrub_frame,
            text="00:00:00 / 00:00:00",
            font=("Arial", 12)
        )
        self.time_label.pack(side="left", padx=5)
        
        # Add scrub slider
        self.scrub_slider = ctk.CTkSlider(
            self.scrub_frame,
            from_=0,
            to=100,
            command=self.on_scrub,
            state="disabled"
        )
        self.scrub_slider.pack(side="left", fill="x", expand=True, padx=5)
        
        # Create preview frame with padding
        self.preview_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color="gray20"
        )
        self.preview_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Add preview canvas with padding
        self.preview_canvas = tk.Canvas(
            self.preview_frame,
            bg="black",
            highlightthickness=0
        )
        self.preview_canvas.pack(fill="both", expand=True, padx=2, pady=(10, 2))
        
        # Add playback controls frame
        self.playback_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color="transparent",
            corner_radius=10,
            border_width=1,
            border_color="gray40"
        )
        self.playback_frame.pack(fill="x", padx=10, pady=5)
        
        # Create centered frame for playback buttons
        self.playback_buttons = ctk.CTkFrame(
            self.playback_frame,
            fg_color="transparent"
        )
        self.playback_buttons.pack(pady=5)
        
        # Add playback buttons with consistent size
        button_width = 45
        button_height = 32
        
        self.backward_frame_button = ctk.CTkButton(
            self.playback_buttons,
            text="⏪ -1",
            command=self.backward_frame,
            width=button_width,
            height=button_height,
            state="disabled"
        )
        self.backward_frame_button.pack(side="left", padx=2)
        
        self.backward_button = ctk.CTkButton(
            self.playback_buttons,
            text="⏪ -10s",
            command=self.backward_10s,
            width=button_width,
            height=button_height,
            state="disabled"
        )
        self.backward_button.pack(side="left", padx=2)
        
        # Create custom play/stop button
        self.play_button = ctk.CTkButton(
            self.playback_buttons,
            text="▶",  # Play symbol
            command=self.toggle_playback,
            width=button_width,
            height=button_height,
            state="disabled",
            fg_color="#1f6aa5",
            text_color="white",
            hover_color="#2b7fc2",
            font=("Arial", 24)
        )
        self.play_button.pack(side="left", padx=2)
        
        self.forward_button = ctk.CTkButton(
            self.playback_buttons,
            text="+10s ⏩",
            command=self.forward_10s,
            width=button_width,
            height=button_height,
            state="disabled"
        )
        self.forward_button.pack(side="left", padx=2)
        
        self.forward_frame_button = ctk.CTkButton(
            self.playback_buttons,
            text="+1 ⏩",
            command=self.forward_frame,
            width=button_width,
            height=button_height,
            state="disabled"
        )
        self.forward_frame_button.pack(side="left", padx=2)
        
        # Add timeline frame
        self.timeline_frame = ctk.CTkFrame(self.main_frame)
        self.timeline_frame.pack(fill="x", padx=5, pady=5)
        
        # Add timeline canvas
        self.timeline_canvas = tk.Canvas(
            self.timeline_frame,
            height=50,
            bg="gray20",
            highlightthickness=0
        )
        self.timeline_canvas.pack(fill="x", expand=True)
        
        # Add analysis progress label
        self.analysis_progress_label = ctk.CTkLabel(
            self.main_frame,
            text="",
            font=("Arial", 14, "bold")
        )
        self.analysis_progress_label.pack(side="bottom", pady=2)
        
        # Add status label with minimal padding
        self.status_label = ctk.CTkLabel(
            self.main_frame,
            text="Ready",
            font=("Arial", 12)
        )
        self.status_label.pack(side="bottom", pady=(2, 0))  # Minimal bottom padding
        
        # Add progress bar with minimal padding
        self.progress_bar = ctk.CTkProgressBar(self.main_frame)
        self.progress_bar.pack(side="bottom", fill="x", padx=10, pady=(2, 0))  # Minimal bottom padding
        
        # Add stats label with minimal padding
        self.stats_label = ctk.CTkLabel(
            self.main_frame,
            text="",
            font=("Arial", 10)
        )
        self.stats_label.pack(side="bottom", pady=(2, 0))  # Minimal bottom padding
        
        # Bind events
        self.preview_canvas.bind("<Button-1>", self.on_canvas_click)
        self.preview_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.timeline_canvas.bind("<Button-1>", self.on_timeline_click)
        
        # Initialize crop variables
        self.crop_rect = None
        self.crop_start = None
        self.crop_mask = None
        self.original_frame = None
        self.is_cropping = False
        self.pending_crop = None  # Store pending crop settings
        
        # Update stats periodically
        self.update_stats()
        
        # Configure window to remove extra space
        self.update_idletasks()  # Update window geometry
        window_height = self.winfo_height()
        self.geometry(f"{self.winfo_width()}x{window_height-50}")  # Reduce height by 50 pixels
        
    def update_stats(self):
        """Update statistics display"""
        stats = self.editor.stats
        stats_text = (
            f"Videos: {stats.total_videos} | "
            f"Segments: {stats.total_segments} | "
            f"Frames: {stats.total_frames} | "
            f"Avg Confidence: {stats.average_confidence:.2f} | "
            f"Errors: {stats.error_count}"
        )
        self.stats_label.configure(text=stats_text)
        self.after(1000, self.update_stats)
        
    def start_analysis(self, mode: str):
        """Start video analysis in the specified mode"""
        if not self.current_video or mode == "Select Analysis Mode":
            return
            
        # Disable analysis dropdown during processing
        self.analysis_dropdown.configure(state="disabled")
        
        # Reset progress
        self.analysis_progress = 0
        self.update_analysis_progress()
        
        # Start analysis in a separate thread
        thread = threading.Thread(
            target=self.run_analysis,
            args=(mode,),
            daemon=True
        )
        thread.start()
        
    def run_analysis(self, mode: str):
        """Run video analysis in background thread"""
        try:
            # Simulate analysis progress
            total_steps = 100
            for i in range(total_steps + 1):
                # Update progress
                self.analysis_progress = i
                self.after(0, self.update_analysis_progress)
                
                # Simulate processing time
                time.sleep(0.1)
                
            # Analysis complete
            self.after(0, self.analysis_complete)
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            self.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
            self.after(0, self.analysis_complete)
            
    def update_analysis_progress(self):
        """Update analysis progress display"""
        if self.analysis_progress > 0:
            self.analysis_progress_label.configure(
                text=f"Analysis Progress: {self.analysis_progress}%"
            )
            self.progress_bar.set(self.analysis_progress / 100)
        else:
            self.analysis_progress_label.configure(text="")
            self.progress_bar.set(0)
            
    def analysis_complete(self):
        """Handle analysis completion"""
        # Re-enable analysis dropdown
        self.analysis_dropdown.configure(state="normal")
        
        # Reset progress display
        self.analysis_progress = 0
        self.update_analysis_progress()
        
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
                    
                # Close previous video if open
                if self.video_cap is not None:
                    self.video_cap.release()
                    
                # Open new video
                self.video_cap = cv2.VideoCapture(str(video_path))
                if not self.video_cap.isOpened():
                    messagebox.showerror("Error", f"Cannot open video file: {filename}")
                    return
                    
                # Get video properties
                self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.duration = self.total_frames / self.fps
                
                # Store the path
                self.current_video = str(video_path)
                self.file_label.configure(
                    text=video_path.name,
                    font=("Arial", 11)
                )
                
                # Enable all controls
                self.analysis_dropdown.configure(state="normal")
                self.split_button.configure(state="normal")
                self.crop_button.configure(state="normal")
                self.delete_button.configure(state="disabled")
                self.save_button.configure(state="normal")
                self.play_button.configure(state="normal")
                self.forward_button.configure(state="normal")
                self.backward_button.configure(state="normal")
                self.forward_frame_button.configure(state="normal")
                self.backward_frame_button.configure(state="normal")
                self.scrub_slider.configure(state="normal")
                
                # Reset playback state
                self.is_playing = False
                self.current_time = 0
                self.update_time_display()
                self.scrub_slider.set(0)
                
                # Show first frame
                ret, frame = self.video_cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.show_frame(frame)
                
                # Create initial segment
                metadata = VideoMetadata(
                    filename=video_path.name,
                    start_time=0,
                    end_time=self.duration,
                    tags=[],
                    summary="Full video",
                    confidence=1.0
                )
                self.current_segments = [VideoSegment(metadata=metadata)]
                self.current_segment = self.current_segments[0]
                
                # Draw timeline
                self.draw_timeline()
                
        except Exception as e:
            logger.error(f"Error selecting video: {str(e)}")
            messagebox.showerror("Error", f"Error selecting video: {str(e)}")
            
    def format_time(self, seconds: float) -> str:
        """Format time in seconds to HH:MM:SS"""
        return str(timedelta(seconds=int(seconds)))
        
    def update_time_display(self):
        """Update time display label"""
        if self.duration > 0:
            current = self.format_time(self.current_time)
            total = self.format_time(self.duration)
            self.time_label.configure(text=f"{current} / {total}")
            
    def toggle_playback(self):
        """Toggle video playback"""
        if not self.current_video:
            return
            
        self.is_playing = not self.is_playing
        
        # Update button appearance
        if self.is_playing:
            self.play_button.configure(
                text="⏹",  # Stop symbol
                fg_color="#c42b1c",
                text_color="white",
                hover_color="#d63b2c",
                font=("Arial", 24)  # Keep large font for stop symbol
            )
        else:
            self.play_button.configure(
                text="▶",  # Play symbol
                fg_color="#1f6aa5",
                text_color="white",
                hover_color="#2b7fc2",
                font=("Arial", 24)  # Keep large font for play symbol
            )
        
        if self.is_playing:
            self.play_video()
            
    def play_video(self):
        """Play video frames"""
        if not self.is_playing or not self.current_video:
            return
            
        try:
            # Read next frame
            ret, frame = self.video_cap.read()
            if ret:
                # Update current time
                self.current_time = self.video_cap.get(cv2.CAP_PROP_POS_FRAMES) / self.fps
                
                # Update display
                self.update_time_display()
                self.scrub_slider.set((self.current_time / self.duration) * 100)
                
                # Convert frame to RGB and display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.show_frame(frame)
                
                # Schedule next frame
                self.after(int(1000 / self.fps), self.play_video)
            else:
                # End of video
                self.is_playing = False
                self.play_button.configure(text="▶️ Play")
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_time = 0
                self.update_time_display()
                self.scrub_slider.set(0)
                
        except Exception as e:
            logger.error(f"Error playing video: {str(e)}")
            self.is_playing = False
            self.play_button.configure(text="▶️ Play")
            
    def show_frame(self, frame):
        """Display frame in preview canvas"""
        try:
            # Get canvas dimensions
            preview_width = self.preview_canvas.winfo_width()
            preview_height = self.preview_canvas.winfo_height()
            
            if preview_width > 0 and preview_height > 0:
                # If we have a pending crop, show it
                if self.pending_crop:
                    crop = self.pending_crop
                    frame = frame[crop['y']:crop['y']+crop['height'], 
                                crop['x']:crop['x']+crop['width']]
                # Otherwise use existing crop if any
                elif self.current_segment and self.current_segment.metadata.crop:
                    crop = self.current_segment.metadata.crop
                    frame = frame[crop['y']:crop['y']+crop['height'], 
                                crop['x']:crop['x']+crop['width']]
                
                # Calculate aspect ratio
                aspect = frame.shape[1] / frame.shape[0]
                
                # Calculate new dimensions with padding
                padding = 4  # 2px padding on each side
                max_width = preview_width - padding
                max_height = preview_height - padding
                
                # Calculate new dimensions
                if max_width / aspect <= max_height:
                    new_width = max_width
                    new_height = int(max_width / aspect)
                else:
                    new_height = max_height
                    new_width = int(max_height * aspect)
                    
                # Resize frame
                frame = cv2.resize(frame, (new_width, new_height))
                
            # Convert to PIL Image and then to PhotoImage
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)
            
            # Update canvas
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(
                preview_width // 2,
                preview_height // 2,
                image=photo,
                anchor="center"
            )
            self.preview_canvas.image = photo  # Keep reference
            
        except Exception as e:
            logger.error(f"Error showing frame: {str(e)}")
            
    def on_scrub(self, value):
        """Handle scrub slider movement"""
        if not self.current_video:
            return
            
        # Calculate new time
        new_time = (value / 100) * self.duration
        
        # Update video position
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, new_time * self.fps)
        self.current_time = new_time
        
        # Update display
        self.update_time_display()
        
        # Show frame at new position
        ret, frame = self.video_cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.show_frame(frame)
            
    def forward_10s(self):
        """Skip forward 10 seconds"""
        if not self.current_video:
            return
            
        new_time = min(self.current_time + 10, self.duration)
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, new_time * self.fps)
        self.current_time = new_time
        self.update_time_display()
        self.scrub_slider.set((new_time / self.duration) * 100)
        
        # Show frame at new position
        ret, frame = self.video_cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.show_frame(frame)
            
    def backward_10s(self):
        """Skip backward 10 seconds"""
        if not self.current_video:
            return
            
        new_time = max(self.current_time - 10, 0)
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, new_time * self.fps)
        self.current_time = new_time
        self.update_time_display()
        self.scrub_slider.set((new_time / self.duration) * 100)
        
        # Show frame at new position
        ret, frame = self.video_cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.show_frame(frame)
            
    def draw_timeline(self):
        """Draw timeline with segments"""
        try:
            if not self.current_video:
                return
                
            # Clear canvas
            self.timeline_canvas.delete("all")
            
            # Get canvas dimensions
            width = self.timeline_canvas.winfo_width()
            height = self.timeline_canvas.winfo_height()
            
            if width <= 1 or height <= 1:
                return
                
            # Draw segments
            duration = self.duration
            for segment in self.current_segments:
                if segment.metadata.is_deleted:
                    continue
                    
                # Calculate segment position
                x1 = (segment.metadata.start_time / duration) * width
                x2 = (segment.metadata.end_time / duration) * width
                
                # Draw segment rectangle
                self.timeline_canvas.create_rectangle(
                    x1, 5, x2, height - 5,
                    fill="gray40",
                    outline="white"
                )
                
                # Draw segment label
                self.timeline_canvas.create_text(
                    (x1 + x2) // 2,
                    height // 2,
                    text=f"{segment.metadata.start_time:.1f}s - {segment.metadata.end_time:.1f}s",
                    fill="white"
                )
                
        except Exception as e:
            logger.error(f"Error drawing timeline: {str(e)}")
            
    def start_edit(self, action: EditAction):
        """Start editing operation"""
        if not self.current_video or not self.current_segment:
            return
            
        self.is_editing = True
        
        if action == EditAction.SPLIT:
            # Get current time from video position
            time = self.current_time
            
            # Show confirmation dialog with time
            if messagebox.askyesno(
                "Confirm Split",
                f"Split video at {self.format_time(time)}?\n\n"
                "This will create two segments:\n"
                f"1. {self.format_time(self.current_segment.metadata.start_time)} to {self.format_time(time)}\n"
                f"2. {self.format_time(time)} to {self.format_time(self.current_segment.metadata.end_time)}"
            ):
                # Split segment
                segment1, segment2 = self.editor.split_segment(
                    self.current_video,
                    time
                )
                
                # Update segments
                idx = self.current_segments.index(self.current_segment)
                self.current_segments[idx] = segment1
                self.current_segments.insert(idx + 1, segment2)
                self.current_segment = segment1
                
                # Redraw timeline
                self.draw_timeline()
                
                # Show success message
                messagebox.showinfo(
                    "Split Complete",
                    f"Video split at {self.format_time(time)}\n\n"
                    f"Segment 1: {self.format_time(segment1.metadata.start_time)} to {self.format_time(segment1.metadata.end_time)}\n"
                    f"Segment 2: {self.format_time(segment2.metadata.start_time)} to {self.format_time(segment2.metadata.end_time)}"
                )
                
        elif action == EditAction.CROP:
            # Enable crop mode
            self.is_cropping = True
            self.preview_canvas.config(cursor="crosshair")
            self.status_label.configure(text="Click and drag to select crop area")
            
        elif action == EditAction.DELETE:
            # Delete current segment
            self.editor.delete_segment(self.current_segment)
            
            # Update segments
            self.current_segments.remove(self.current_segment)
            if self.current_segments:
                self.current_segment = self.current_segments[0]
            else:
                self.current_segment = None
                
            # Redraw timeline
            self.draw_timeline()
            
        elif action == EditAction.SAVE:
            # Apply pending crop if exists
            if self.pending_crop:
                self.current_segment.metadata.crop = self.pending_crop
                self.pending_crop = None
                
            # Save current segment
            try:
                output_path, metadata_path = self.editor.save_segment(
                    self.current_video,
                    self.current_segment
                )
                
                messagebox.showinfo(
                    "Success",
                    f"Segment saved to:\n{output_path}\nMetadata: {metadata_path}"
                )
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save segment: {str(e)}")
                
        self.is_editing = False
        
    def get_current_time(self) -> Optional[float]:
        """Get current time from timeline click position"""
        try:
            # Get click position
            x = self.timeline_canvas.winfo_pointerx() - self.timeline_canvas.winfo_rootx()
            
            # Convert to time
            width = self.timeline_canvas.winfo_width()
            duration = self.duration
            time = (x / width) * duration
            
            return time
            
        except Exception as e:
            logger.error(f"Error getting current time: {str(e)}")
            return None
            
    def on_canvas_click(self, event):
        """Handle canvas click"""
        if not self.is_cropping:
            return
            
        # Store original frame if not already stored
        if self.original_frame is None:
            self.original_frame = self.preview_canvas.image
            
        # Start crop rectangle
        self.crop_start = (event.x, event.y)
        self.crop_rect = self.preview_canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="red",
            width=2
        )
        
    def on_canvas_drag(self, event):
        """Handle canvas drag"""
        if not self.is_cropping or not self.crop_rect or not self.crop_start:
            return
            
        # Update crop rectangle
        self.preview_canvas.coords(
            self.crop_rect,
            self.crop_start[0],
            self.crop_start[1],
            event.x,
            event.y
        )
        
    def on_canvas_release(self, event):
        """Handle canvas release"""
        if not self.is_cropping or not self.crop_rect or not self.crop_start:
            return
            
        # Get crop coordinates
        x1, y1, x2, y2 = self.preview_canvas.coords(self.crop_rect)
        
        # Create mask for the cropped area
        if self.original_frame:
            # Convert PhotoImage to PIL Image
            pil_image = ImageTk.getimage(self.original_frame)
            
            # Create a copy for the mask
            mask = pil_image.copy()
            
            # Create a semi-transparent overlay
            overlay = Image.new('RGBA', mask.size, (0, 0, 0, 192))  # 75% opacity
            
            # Create a mask for the crop area
            crop_mask = Image.new('L', mask.size, 0)
            crop_draw = ImageDraw.Draw(crop_mask)
            crop_draw.rectangle([x1, y1, x2, y2], fill=255)
            
            # Apply the overlay to the masked area
            mask.paste(overlay, (0, 0), crop_mask)
            
            # Convert back to PhotoImage
            self.crop_mask = ImageTk.PhotoImage(mask)
            
            # Update canvas with masked image
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(
                self.preview_canvas.winfo_width() // 2,
                self.preview_canvas.winfo_height() // 2,
                image=self.crop_mask,
                anchor="center"
            )
            
            # Show confirmation dialog
            if messagebox.askyesno("Confirm Crop", "Do you want to apply this crop?"):
                # Convert to video coordinates
                preview_width = self.preview_canvas.winfo_width()
                preview_height = self.preview_canvas.winfo_height()
                
                # Get video dimensions
                cap = cv2.VideoCapture(str(self.current_video))
                video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                # Calculate scale factors
                scale_x = video_width / preview_width
                scale_y = video_height / preview_height
                
                # Convert coordinates
                x = int(min(x1, x2) * scale_x)
                y = int(min(y1, y2) * scale_y)
                width = int(abs(x2 - x1) * scale_x)
                height = int(abs(y2 - y1) * scale_y)
                
                # Store pending crop settings
                self.pending_crop = {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height
                }
                
                # Show preview of crop
                self.center_crop_preview(x, y, width, height)
            else:
                # Reset to original frame
                self.preview_canvas.delete("all")
                self.preview_canvas.create_image(
                    self.preview_canvas.winfo_width() // 2,
                    self.preview_canvas.winfo_height() // 2,
                    image=self.original_frame,
                    anchor="center"
                )
        
        # Reset crop mode
        self.preview_canvas.config(cursor="")
        self.crop_rect = None
        self.crop_start = None
        self.crop_mask = None
        self.original_frame = None
        self.is_cropping = False
        self.status_label.configure(text="Ready")
        
    def center_crop_preview(self, x: int, y: int, width: int, height: int):
        """Center and fill the preview with cropped area"""
        try:
            # Get current frame
            ret, frame = self.video_cap.read()
            if not ret:
                return
                
            # Apply crop
            cropped = frame[y:y+height, x:x+width]
            
            # Get preview dimensions
            preview_width = self.preview_canvas.winfo_width()
            preview_height = self.preview_canvas.winfo_height()
            
            # Calculate aspect ratio
            aspect = width / height
            
            # Calculate new dimensions with padding
            padding = 4  # 2px padding on each side
            max_width = preview_width - padding
            max_height = preview_height - padding
            
            # Calculate new dimensions to fill preview
            if max_width / aspect <= max_height:
                new_height = max_height
                new_width = int(max_height * aspect)
            else:
                new_width = max_width
                new_height = int(max_width / aspect)
                
            # Resize cropped frame
            resized = cv2.resize(cropped, (new_width, new_height))
            
            # Convert to RGB
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and then to PhotoImage
            image = Image.fromarray(resized)
            photo = ImageTk.PhotoImage(image)
            
            # Update canvas
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(
                preview_width // 2,
                preview_height // 2,
                image=photo,
                anchor="center"
            )
            self.preview_canvas.image = photo  # Keep reference
            
        except Exception as e:
            logger.error(f"Error centering crop preview: {str(e)}")
            
    def on_timeline_click(self, event):
        """Handle timeline click"""
        if not self.current_video:
            return
            
        # Get click time
        time = self.get_current_time()
        if time is None:
            return
            
        # Find segment containing time
        for segment in self.current_segments:
            if segment.metadata.start_time <= time <= segment.metadata.end_time:
                self.current_segment = segment
                self.show_frame(self.editor.get_segment_preview(self.current_video, time))
                break
                
    def forward_frame(self):
        """Move forward one frame"""
        if not self.current_video:
            return
            
        # Get current frame number
        current_frame = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Move to next frame
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + 1)
        
        # Update time
        self.current_time = (current_frame + 1) / self.fps
        self.update_time_display()
        self.scrub_slider.set((self.current_time / self.duration) * 100)
        
        # Show frame
        ret, frame = self.video_cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.show_frame(frame)
            
    def backward_frame(self):
        """Move backward one frame"""
        if not self.current_video:
            return
            
        # Get current frame number
        current_frame = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Move to previous frame (ensure we don't go below 0)
        new_frame = max(0, current_frame - 1)
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        
        # Update time
        self.current_time = new_frame / self.fps
        self.update_time_display()
        self.scrub_slider.set((self.current_time / self.duration) * 100)
        
        # Show frame
        ret, frame = self.video_cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.show_frame(frame)
            
    def __del__(self):
        """Cleanup when window is closed"""
        if self.video_cap is not None:
            self.video_cap.release()

    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def show_tooltip(event):
            # Remove any existing tooltip
            if self.active_tooltip:
                self.active_tooltip.destroy()
                
            # Create tooltip window
            tooltip = tk.Toplevel(self)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root}+{event.y_root-30}")  # 10px above + 20px for tooltip height
            
            # Create tooltip label
            label = tk.Label(
                tooltip,
                text=text,
                justify="left",
                background="#ffffe0",
                relief="solid",
                borderwidth=1,
                font=("Arial", 10)
            )
            label.pack()
            
            # Store tooltip reference
            self.active_tooltip = tooltip
            
        def hide_tooltip(event):
            if self.active_tooltip:
                self.active_tooltip.destroy()
                self.active_tooltip = None
        
        # Bind events
        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)
        
def main():
    setup_logging()
    app = VideoEditorGUI()
    app.mainloop()
    
if __name__ == "__main__":
    main() 