"""
Unified Media Player with Video Editing and Analysis capabilities
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
from ..processors.video_analyzer import VideoAnalyzer, AnalysisType
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

class UnifiedMediaPlayer(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Initialize tooltip tracking
        self.active_tooltip = None
        
        # Configure window
        self.title("PowerCoreAi Media Player")
        self.geometry("1200x800")
        self.minsize(1200, 800)
        
        # Initialize components
        self.config = ProcessingConfig(
            output_dir=Path.home() / ".powercoreai" / "output",
            frames_dir=Path.home() / ".powercoreai" / "output" / "frames"
        )
        self.editor = VideoEditor(config=self.config)
        self.analyzer = VideoAnalyzer()
        
        # Video state
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
        
        # Audio state
        self.audio_volume = 1.0
        self.is_muted = False
        
        # Filter state
        self.current_filter = None
        self.filter_params = {}
        
        # Create main layout
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Create main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create top toolbar
        self.toolbar = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.toolbar.pack(fill="x", padx=5, pady=5)
        
        # File operations
        self.file_frame = ctk.CTkFrame(self.toolbar, fg_color="transparent")
        self.file_frame.pack(side="left", padx=5)
        
        self.select_button = ctk.CTkButton(
            self.file_frame,
            text="Open Media",
            command=self.select_media,
            width=100
        )
        self.select_button.pack(side="left", padx=5)
        
        self.file_label = ctk.CTkLabel(
            self.file_frame,
            text="No media selected",
            wraplength=200
        )
        self.file_label.pack(side="left", padx=5)
        
        # Analysis controls
        self.analysis_frame = ctk.CTkFrame(self.toolbar, fg_color="transparent")
        self.analysis_frame.pack(side="left", padx=5)
        
        self.analysis_type = ctk.CTkOptionMenu(
            self.analysis_frame,
            values=[t.value for t in AnalysisType],
            command=self.on_analysis_type_change
        )
        self.analysis_type.set(AnalysisType.BASIC.value)
        self.analysis_type.pack(side="left", padx=5)
        
        self.analyze_button = ctk.CTkButton(
            self.analysis_frame,
            text="Analyze",
            command=self.analyze_video,
            state="disabled",
            width=100
        )
        self.analyze_button.pack(side="left", padx=5)
        
        # Edit controls
        self.edit_frame = ctk.CTkFrame(self.toolbar, fg_color="transparent")
        self.edit_frame.pack(side="right", padx=5)
        
        # Basic edit controls
        self.split_button = ctk.CTkButton(
            self.edit_frame,
            text="✂ Split",
            command=lambda: self.start_edit(EditAction.SPLIT),
            state="disabled",
            width=80
        )
        self.split_button.pack(side="left", padx=2)
        
        self.crop_button = ctk.CTkButton(
            self.edit_frame,
            text="⬛ Crop",
            command=lambda: self.start_edit(EditAction.CROP),
            state="disabled",
            width=80
        )
        self.crop_button.pack(side="left", padx=2)
        
        self.delete_button = ctk.CTkButton(
            self.edit_frame,
            text="❌ Delete",
            command=lambda: self.start_edit(EditAction.DELETE),
            state="disabled",
            width=80
        )
        self.delete_button.pack(side="left", padx=2)
        
        # Advanced edit controls
        self.filter_button = ctk.CTkButton(
            self.edit_frame,
            text="🎨 Filter",
            command=self.show_filter_dialog,
            state="disabled",
            width=80
        )
        self.filter_button.pack(side="left", padx=2)
        
        self.transition_button = ctk.CTkButton(
            self.edit_frame,
            text="🔄 Transition",
            command=self.show_transition_dialog,
            state="disabled",
            width=80
        )
        self.transition_button.pack(side="left", padx=2)
        
        self.text_button = ctk.CTkButton(
            self.edit_frame,
            text="📝 Text",
            command=self.show_text_dialog,
            state="disabled",
            width=80
        )
        self.text_button.pack(side="left", padx=2)
        
        self.save_button = ctk.CTkButton(
            self.edit_frame,
            text="💾 Save",
            command=lambda: self.start_edit(EditAction.SAVE),
            state="disabled",
            width=80
        )
        self.save_button.pack(side="left", padx=2)
        
        # Create preview area
        self.preview_frame = ctk.CTkFrame(self.main_frame)
        self.preview_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.preview_label = ctk.CTkLabel(
            self.preview_frame,
            text="Select a media file to begin",
            font=("Arial", 14)
        )
        self.preview_label.pack(expand=True)
        
        # Create playback controls
        self.playback_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.playback_frame.pack(fill="x", padx=5, pady=5)
        
        # Time display
        self.time_label = ctk.CTkLabel(
            self.playback_frame,
            text="00:00:00 / 00:00:00",
            font=("Arial", 12)
        )
        self.time_label.pack(side="left", padx=5)
        
        # Playback buttons
        self.backward_button = ctk.CTkButton(
            self.playback_frame,
            text="⏪",
            command=self.backward_10s,
            state="disabled",
            width=40
        )
        self.backward_button.pack(side="left", padx=2)
        
        self.play_button = ctk.CTkButton(
            self.playback_frame,
            text="▶",
            command=self.toggle_playback,
            state="disabled",
            width=40
        )
        self.play_button.pack(side="left", padx=2)
        
        self.forward_button = ctk.CTkButton(
            self.playback_frame,
            text="⏩",
            command=self.forward_10s,
            state="disabled",
            width=40
        )
        self.forward_button.pack(side="left", padx=2)
        
        # Audio controls
        self.volume_button = ctk.CTkButton(
            self.playback_frame,
            text="🔊",
            command=self.toggle_mute,
            state="disabled",
            width=40
        )
        self.volume_button.pack(side="left", padx=2)
        
        self.volume_slider = ctk.CTkSlider(
            self.playback_frame,
            from_=0,
            to=1,
            command=self.set_volume,
            state="disabled",
            width=100
        )
        self.volume_slider.set(1.0)
        self.volume_slider.pack(side="left", padx=5)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.playback_frame)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=10)
        self.progress_bar.set(0)
        
        # Status bar
        self.status_label = ctk.CTkLabel(
            self.main_frame,
            text="Ready",
            font=("Arial", 12)
        )
        self.status_label.pack(side="bottom", pady=5)
        
    def select_media(self):
        """Open file dialog to select media file"""
        filetypes = [
            ("Media files", "*.mp4 *.avi *.mkv *.mov *.jpg *.jpeg *.png"),
            ("All files", "*.*")
        ]
        
        try:
            initial_dir = None
            if self.current_video:
                initial_dir = str(Path(self.current_video).parent)
            
            filename = filedialog.askopenfilename(
                title="Select Media File",
                filetypes=filetypes,
                initialdir=initial_dir
            )
            
            if filename:
                media_path = Path(filename).resolve()
                if not media_path.exists():
                    messagebox.showerror("Error", f"File not found: {filename}")
                    return
                
                # Handle video files
                if media_path.suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov']:
                    try:
                        cap = cv2.VideoCapture(str(media_path))
                        if not cap.isOpened():
                            messagebox.showerror("Error", f"Cannot open video file: {filename}")
                            return
                        
                        # Store video properties
                        self.fps = cap.get(cv2.CAP_PROP_FPS)
                        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        self.duration = self.total_frames / self.fps
                        cap.release()
                        
                        # Update UI
                        self.current_video = str(media_path)
                        self.file_label.configure(text=media_path.name)
                        self.analyze_button.configure(state="normal")
                        self.play_button.configure(state="normal")
                        self.backward_button.configure(state="normal")
                        self.forward_button.configure(state="normal")
                        self.status_label.configure(text=f"Loaded: {media_path.name}")
                        
                        # Show preview
                        self.show_preview_frame(media_path)
                        
                    except Exception as e:
                        messagebox.showerror("Error", f"Error testing video file: {str(e)}")
                        return
                
                # Handle image files
                elif media_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    try:
                        image = Image.open(media_path)
                        self.current_video = str(media_path)
                        self.file_label.configure(text=media_path.name)
                        self.status_label.configure(text=f"Loaded: {media_path.name}")
                        self.show_image_preview(image)
                    except Exception as e:
                        messagebox.showerror("Error", f"Error loading image: {str(e)}")
                        return
                
        except Exception as e:
            logger.error(f"Error selecting media: {str(e)}")
            messagebox.showerror("Error", f"Error selecting media: {str(e)}")
    
    def show_preview_frame(self, video_path: Path):
        """Show first frame of the video as preview"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview_width = self.preview_frame.winfo_width() - 20
                preview_height = self.preview_frame.winfo_height() - 20
                
                if preview_width > 0 and preview_height > 0:
                    aspect = frame.shape[1] / frame.shape[0]
                    if preview_width / aspect <= preview_height:
                        new_width = preview_width
                        new_height = int(preview_width / aspect)
                    else:
                        new_height = preview_height
                        new_width = int(preview_height * aspect)
                    
                    frame = cv2.resize(frame, (new_width, new_height))
                
                image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image)
                self.preview_label.configure(image=photo, text="")
                self.preview_label.image = photo
            else:
                self.preview_label.configure(text="Could not load video preview")
                
        except Exception as e:
            logger.error(f"Error showing preview: {str(e)}")
            self.preview_label.configure(text="Error loading preview")
    
    def show_image_preview(self, image: Image.Image):
        """Show image preview"""
        try:
            preview_width = self.preview_frame.winfo_width() - 20
            preview_height = self.preview_frame.winfo_height() - 20
            
            if preview_width > 0 and preview_height > 0:
                aspect = image.width / image.height
                if preview_width / aspect <= preview_height:
                    new_width = preview_width
                    new_height = int(preview_width / aspect)
                else:
                    new_height = preview_height
                    new_width = int(preview_height * aspect)
                
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=photo, text="")
            self.preview_label.image = photo
            
        except Exception as e:
            logger.error(f"Error showing image preview: {str(e)}")
            self.preview_label.configure(text="Error loading image preview")
    
    def on_analysis_type_change(self, choice):
        """Handle analysis type change"""
        self.analysis_type.set(choice)
    
    def analyze_video(self):
        """Start video analysis"""
        if not self.current_video:
            return
            
        if self.is_analyzing:
            messagebox.showwarning("Warning", "Analysis already in progress")
            return
            
        self.is_analyzing = True
        self.analyze_button.configure(state="disabled")
        self.status_label.configure(text="Analyzing...")
        
        # Start analysis in separate thread
        analysis_thread = threading.Thread(
            target=self.run_analysis,
            args=(60, AnalysisType(self.analysis_type.get())),
            daemon=True
        )
        analysis_thread.start()
    
    def run_analysis(self, interval: int, analysis_type: AnalysisType):
        """Run video analysis"""
        try:
            result = self.analyzer.analyze(
                self.current_video,
                interval=interval,
                analysis_type=analysis_type
            )
            
            # Update UI in main thread
            self.after(0, self.analysis_complete, result.manifest_path, result.grid_path)
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            self.after(0, self.analysis_failed, str(e))
    
    def analysis_complete(self, manifest_path: str, grid_path: str = None):
        """Handle analysis completion"""
        self.is_analyzing = False
        self.analyze_button.configure(state="normal")
        self.status_label.configure(text="Analysis complete")
        
        # Show results
        if grid_path:
            try:
                grid_image = Image.open(grid_path)
                self.show_image_preview(grid_image)
            except Exception as e:
                logger.error(f"Error showing analysis grid: {str(e)}")
    
    def analysis_failed(self, error_msg: str):
        """Handle analysis failure"""
        self.is_analyzing = False
        self.analyze_button.configure(state="normal")
        self.status_label.configure(text="Analysis failed")
        messagebox.showerror("Analysis Error", error_msg)
    
    def toggle_playback(self):
        """Toggle video playback"""
        if not self.current_video:
            return
            
        if self.is_playing:
            self.is_playing = False
            self.play_button.configure(text="▶")
        else:
            self.is_playing = True
            self.play_button.configure(text="⏸")
            self.play_video()
    
    def play_video(self):
        """Play video frames"""
        if not self.is_playing or not self.current_video:
            return
            
        try:
            if not self.video_cap:
                self.video_cap = cv2.VideoCapture(self.current_video)
            
            ret, frame = self.video_cap.read()
            if ret:
                # Update current time
                self.current_time = self.video_cap.get(cv2.CAP_PROP_POS_FRAMES) / self.fps
                self.update_time_display()
                
                # Show frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image)
                self.preview_label.configure(image=photo, text="")
                self.preview_label.image = photo
                
                # Schedule next frame
                self.after(int(1000/self.fps), self.play_video)
            else:
                # End of video
                self.video_cap.release()
                self.video_cap = None
                self.is_playing = False
                self.play_button.configure(text="▶")
                self.current_time = 0
                self.update_time_display()
                
        except Exception as e:
            logger.error(f"Error playing video: {str(e)}")
            self.is_playing = False
            self.play_button.configure(text="▶")
    
    def update_time_display(self):
        """Update time display"""
        current = timedelta(seconds=int(self.current_time))
        total = timedelta(seconds=int(self.duration))
        self.time_label.configure(text=f"{current} / {total}")
        self.progress_bar.set(self.current_time / self.duration)
    
    def forward_10s(self):
        """Skip forward 10 seconds"""
        if not self.current_video:
            return
            
        if self.video_cap:
            new_time = min(self.current_time + 10, self.duration)
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, new_time * self.fps)
            self.current_time = new_time
            self.update_time_display()
    
    def backward_10s(self):
        """Skip backward 10 seconds"""
        if not self.current_video:
            return
            
        if self.video_cap:
            new_time = max(self.current_time - 10, 0)
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, new_time * self.fps)
            self.current_time = new_time
            self.update_time_display()
    
    def start_edit(self, action: EditAction):
        """Start video editing operation"""
        if not self.current_video:
            return
            
        if action == EditAction.SPLIT:
            self.split_video()
        elif action == EditAction.CROP:
            self.crop_video()
        elif action == EditAction.DELETE:
            self.delete_segment()
        elif action == EditAction.SAVE:
            self.save_changes()
    
    def split_video(self):
        """Split video at current position"""
        if not self.current_video or not self.video_cap:
            return
            
        try:
            frame_pos = int(self.current_time * self.fps)
            self.editor.split_video(self.current_video, frame_pos)
            self.status_label.configure(text="Video split at current position")
        except Exception as e:
            logger.error(f"Error splitting video: {str(e)}")
            messagebox.showerror("Error", f"Failed to split video: {str(e)}")
    
    def crop_video(self):
        """Crop video"""
        if not self.current_video:
            return
            
        try:
            # Get crop dimensions from user
            crop_dialog = CropDialog(self)
            if crop_dialog.result:
                x, y, width, height = crop_dialog.result
                self.editor.crop_video(self.current_video, x, y, width, height)
                self.status_label.configure(text="Video cropped")
        except Exception as e:
            logger.error(f"Error cropping video: {str(e)}")
            messagebox.showerror("Error", f"Failed to crop video: {str(e)}")
    
    def delete_segment(self):
        """Delete current segment"""
        if not self.current_video or not self.current_segment:
            return
            
        try:
            self.editor.delete_segment(self.current_video, self.current_segment)
            self.status_label.configure(text="Segment deleted")
        except Exception as e:
            logger.error(f"Error deleting segment: {str(e)}")
            messagebox.showerror("Error", f"Failed to delete segment: {str(e)}")
    
    def save_changes(self):
        """Save all changes"""
        if not self.current_video:
            return
            
        try:
            output_path = self.editor.save_changes(self.current_video)
            self.status_label.configure(text=f"Changes saved to: {output_path}")
            messagebox.showinfo("Success", f"Changes saved to:\n{output_path}")
        except Exception as e:
            logger.error(f"Error saving changes: {str(e)}")
            messagebox.showerror("Error", f"Failed to save changes: {str(e)}")
    
    def show_filter_dialog(self):
        """Show filter selection dialog"""
        if not self.current_video:
            return
            
        dialog = FilterDialog(self)
        if dialog.result:
            filter_type, params = dialog.result
            self.current_filter = filter_type
            self.filter_params = params
            self.apply_filter()
    
    def show_transition_dialog(self):
        """Show transition selection dialog"""
        if not self.current_video:
            return
            
        dialog = TransitionDialog(self)
        if dialog.result:
            transition_type, duration = dialog.result
            self.add_transition(transition_type, duration)
    
    def show_text_dialog(self):
        """Show text overlay dialog"""
        if not self.current_video:
            return
            
        dialog = TextOverlayDialog(self)
        if dialog.result:
            text, position, font, color = dialog.result
            self.add_text_overlay(text, position, font, color)
    
    def apply_filter(self):
        """Apply selected filter to video"""
        if not self.current_video or not self.current_filter:
            return
            
        try:
            self.editor.apply_filter(
                self.current_video,
                self.current_filter,
                self.filter_params
            )
            self.status_label.configure(text=f"Applied {self.current_filter} filter")
        except Exception as e:
            logger.error(f"Error applying filter: {str(e)}")
            messagebox.showerror("Error", f"Failed to apply filter: {str(e)}")
    
    def add_transition(self, transition_type: str, duration: float):
        """Add transition effect"""
        if not self.current_video:
            return
            
        try:
            self.editor.add_transition(
                self.current_video,
                transition_type,
                duration
            )
            self.status_label.configure(text=f"Added {transition_type} transition")
        except Exception as e:
            logger.error(f"Error adding transition: {str(e)}")
            messagebox.showerror("Error", f"Failed to add transition: {str(e)}")
    
    def add_text_overlay(self, text: str, position: Tuple[int, int], font: str, color: str):
        """Add text overlay to video"""
        if not self.current_video:
            return
            
        try:
            self.editor.add_text_overlay(
                self.current_video,
                text,
                position,
                font,
                color
            )
            self.status_label.configure(text="Added text overlay")
        except Exception as e:
            logger.error(f"Error adding text overlay: {str(e)}")
            messagebox.showerror("Error", f"Failed to add text overlay: {str(e)}")
    
    def toggle_mute(self):
        """Toggle audio mute"""
        self.is_muted = not self.is_muted
        if self.is_muted:
            self.volume_button.configure(text="🔇")
            self.set_volume(0)
        else:
            self.volume_button.configure(text="🔊")
            self.set_volume(self.audio_volume)
    
    def set_volume(self, value: float):
        """Set audio volume"""
        self.audio_volume = value
        if not self.is_muted:
            self.editor.set_volume(self.current_video, value)
            self.status_label.configure(text=f"Volume: {int(value * 100)}%")
    
    def __del__(self):
        """Cleanup"""
        if self.video_cap:
            self.video_cap.release()

class CropDialog(ctk.CTkToplevel):
    """Dialog for crop dimensions"""
    def __init__(self, parent):
        super().__init__(parent)
        self.result = None
        
        self.title("Crop Video")
        self.geometry("300x200")
        
        # Create input fields
        self.x_label = ctk.CTkLabel(self, text="X:")
        self.x_label.pack(pady=5)
        self.x_entry = ctk.CTkEntry(self)
        self.x_entry.pack(pady=5)
        
        self.y_label = ctk.CTkLabel(self, text="Y:")
        self.y_label.pack(pady=5)
        self.y_entry = ctk.CTkEntry(self)
        self.y_entry.pack(pady=5)
        
        self.width_label = ctk.CTkLabel(self, text="Width:")
        self.width_label.pack(pady=5)
        self.width_entry = ctk.CTkEntry(self)
        self.width_entry.pack(pady=5)
        
        self.height_label = ctk.CTkLabel(self, text="Height:")
        self.height_label.pack(pady=5)
        self.height_entry = ctk.CTkEntry(self)
        self.height_entry.pack(pady=5)
        
        # Add OK button
        self.ok_button = ctk.CTkButton(
            self,
            text="OK",
            command=self.on_ok
        )
        self.ok_button.pack(pady=10)
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        parent.wait_window(self)
    
    def on_ok(self):
        """Handle OK button click"""
        try:
            x = int(self.x_entry.get())
            y = int(self.y_entry.get())
            width = int(self.width_entry.get())
            height = int(self.height_entry.get())
            
            if width <= 0 or height <= 0:
                raise ValueError("Width and height must be positive")
                
            self.result = (x, y, width, height)
            self.destroy()
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")

class FilterDialog(ctk.CTkToplevel):
    """Dialog for selecting video filters"""
    def __init__(self, parent):
        super().__init__(parent)
        self.result = None
        
        self.title("Apply Filter")
        self.geometry("400x300")
        
        # Filter selection
        self.filter_label = ctk.CTkLabel(self, text="Select Filter:")
        self.filter_label.pack(pady=5)
        
        self.filter_type = ctk.CTkOptionMenu(
            self,
            values=[
                "None",
                "Grayscale",
                "Sepia",
                "Blur",
                "Sharpen",
                "Edge Detection",
                "Emboss",
                "Cartoon"
            ]
        )
        self.filter_type.pack(pady=5)
        
        # Filter parameters
        self.params_frame = ctk.CTkFrame(self)
        self.params_frame.pack(fill="x", padx=10, pady=5)
        
        # Intensity slider
        self.intensity_label = ctk.CTkLabel(self.params_frame, text="Intensity:")
        self.intensity_label.pack(pady=5)
        
        self.intensity_slider = ctk.CTkSlider(
            self.params_frame,
            from_=0,
            to=1
        )
        self.intensity_slider.set(0.5)
        self.intensity_slider.pack(pady=5)
        
        # Add OK button
        self.ok_button = ctk.CTkButton(
            self,
            text="Apply",
            command=self.on_ok
        )
        self.ok_button.pack(pady=10)
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        parent.wait_window(self)
    
    def on_ok(self):
        """Handle OK button click"""
        filter_type = self.filter_type.get()
        if filter_type != "None":
            self.result = (filter_type, {"intensity": self.intensity_slider.get()})
        self.destroy()

class TransitionDialog(ctk.CTkToplevel):
    """Dialog for selecting video transitions"""
    def __init__(self, parent):
        super().__init__(parent)
        self.result = None
        
        self.title("Add Transition")
        self.geometry("300x200")
        
        # Transition selection
        self.transition_label = ctk.CTkLabel(self, text="Select Transition:")
        self.transition_label.pack(pady=5)
        
        self.transition_type = ctk.CTkOptionMenu(
            self,
            values=[
                "Fade",
                "Slide",
                "Wipe",
                "Dissolve",
                "Zoom"
            ]
        )
        self.transition_type.pack(pady=5)
        
        # Duration slider
        self.duration_label = ctk.CTkLabel(self, text="Duration (seconds):")
        self.duration_label.pack(pady=5)
        
        self.duration_slider = ctk.CTkSlider(
            self,
            from_=0.5,
            to=3.0
        )
        self.duration_slider.set(1.0)
        self.duration_slider.pack(pady=5)
        
        # Add OK button
        self.ok_button = ctk.CTkButton(
            self,
            text="Add",
            command=self.on_ok
        )
        self.ok_button.pack(pady=10)
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        parent.wait_window(self)
    
    def on_ok(self):
        """Handle OK button click"""
        self.result = (
            self.transition_type.get(),
            self.duration_slider.get()
        )
        self.destroy()

class TextOverlayDialog(ctk.CTkToplevel):
    """Dialog for adding text overlay"""
    def __init__(self, parent):
        super().__init__(parent)
        self.result = None
        
        self.title("Add Text Overlay")
        self.geometry("400x300")
        
        # Text input
        self.text_label = ctk.CTkLabel(self, text="Enter Text:")
        self.text_label.pack(pady=5)
        
        self.text_entry = ctk.CTkEntry(self, width=300)
        self.text_entry.pack(pady=5)
        
        # Font selection
        self.font_label = ctk.CTkLabel(self, text="Font:")
        self.font_label.pack(pady=5)
        
        self.font_entry = ctk.CTkEntry(self, width=200)
        self.font_entry.insert(0, "Arial")
        self.font_entry.pack(pady=5)
        
        # Color selection
        self.color_label = ctk.CTkLabel(self, text="Color:")
        self.color_label.pack(pady=5)
        
        self.color_entry = ctk.CTkEntry(self, width=100)
        self.color_entry.insert(0, "#FFFFFF")
        self.color_entry.pack(pady=5)
        
        # Position selection
        self.position_label = ctk.CTkLabel(self, text="Position:")
        self.position_label.pack(pady=5)
        
        self.position_frame = ctk.CTkFrame(self)
        self.position_frame.pack(pady=5)
        
        self.x_label = ctk.CTkLabel(self.position_frame, text="X:")
        self.x_label.pack(side="left", padx=5)
        
        self.x_entry = ctk.CTkEntry(self.position_frame, width=50)
        self.x_entry.insert(0, "10")
        self.x_entry.pack(side="left", padx=5)
        
        self.y_label = ctk.CTkLabel(self.position_frame, text="Y:")
        self.y_label.pack(side="left", padx=5)
        
        self.y_entry = ctk.CTkEntry(self.position_frame, width=50)
        self.y_entry.insert(0, "10")
        self.y_entry.pack(side="left", padx=5)
        
        # Add OK button
        self.ok_button = ctk.CTkButton(
            self,
            text="Add",
            command=self.on_ok
        )
        self.ok_button.pack(pady=10)
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        parent.wait_window(self)
    
    def on_ok(self):
        """Handle OK button click"""
        try:
            text = self.text_entry.get()
            font = self.font_entry.get()
            color = self.color_entry.get()
            x = int(self.x_entry.get())
            y = int(self.y_entry.get())
            
            self.result = (text, (x, y), font, color)
            self.destroy()
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")

def main():
    """Start the application"""
    app = UnifiedMediaPlayer()
    app.mainloop()

if __name__ == "__main__":
    main() 