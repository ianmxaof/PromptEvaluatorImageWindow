"""
Video editing toolkit for interactive segment manipulation
"""
import os
import subprocess
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
import logging
import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum

from ..models.data_models import (
    VideoMetadata,
    ProcessingConfig,
    VideoSegment,
    ProcessingResult,
    UserConfig,
    ProcessingStats,
    SecurityConfig
)

logger = logging.getLogger(__name__)

class EditAction(Enum):
    SPLIT = "split"
    CROP = "crop"
    DELETE = "delete"
    SAVE = "save"

class VideoEditor:
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize video editor with configuration"""
        self.config = config or ProcessingConfig(
            output_dir=Path.home() / ".organizerbot" / "output",
            frames_dir=Path.home() / ".organizerbot" / "output" / "frames"
        )
        self.stats = ProcessingStats()
        
    def split_segment(self, video_path: str, split_time: float) -> Tuple[VideoSegment, VideoSegment]:
        """Split a video segment at the specified time"""
        try:
            # Get video duration
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            cap.release()
            
            if split_time <= 0 or split_time >= duration:
                raise ValueError(f"Invalid split time: {split_time}")
                
            # Create metadata for segments
            metadata1 = VideoMetadata(
                filename=Path(video_path).name,
                start_time=0,
                end_time=split_time,
                tags=[],
                summary="First segment",
                confidence=1.0
            )
            
            metadata2 = VideoMetadata(
                filename=Path(video_path).name,
                start_time=split_time,
                end_time=duration,
                tags=[],
                summary="Second segment",
                confidence=1.0
            )
            
            # Create segments
            segment1 = VideoSegment(metadata=metadata1)
            segment2 = VideoSegment(metadata=metadata2)
            
            # Update stats
            self.stats.total_segments += 2
            
            return segment1, segment2
            
        except Exception as e:
            logger.error(f"Error splitting segment: {str(e)}")
            raise
            
    def apply_crop(self, video_path: str, crop_params: Dict[str, int], output_path: Optional[str] = None) -> str:
        """Apply crop filter to video segment"""
        try:
            if not output_path:
                output_path = str(self.config.output_dir / f"{Path(video_path).stem}_cropped.mp4")
                
            # Validate crop parameters
            crop = VideoMetadata(crop=crop_params).crop
            if crop is None:
                raise ValueError("Invalid crop parameters")
                
            # Build FFmpeg command with crop filter
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vf', f"crop={crop['width']}:{crop['height']}:{crop['x']}:{crop['y']}",
                '-c:a', 'copy',  # Copy audio stream
                '-y',  # Overwrite output
                output_path
            ]
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr.decode()}")
                raise RuntimeError("Failed to apply crop")
                
            return output_path
            
        except Exception as e:
            logger.error(f"Error applying crop: {str(e)}")
            raise
            
    def save_segment(self, video_path: str, segment: VideoSegment, output_path: Optional[str] = None) -> Tuple[str, str]:
        """Save a video segment with metadata"""
        try:
            if not output_path:
                # Generate output path based on tags
                tag_str = "_".join(segment.metadata.tags) if segment.metadata.tags else "untagged"
                output_path = str(self.config.output_dir / f"{Path(video_path).stem}_{tag_str}.mp4")
                
            # Build FFmpeg command for segment extraction
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-ss', str(segment.metadata.start_time),
                '-to', str(segment.metadata.end_time),
                '-c', 'copy',  # Copy streams without re-encoding
                '-y',  # Overwrite output
                output_path
            ]
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr.decode()}")
                raise RuntimeError("Failed to save segment")
                
            # Apply crop if specified
            if segment.metadata.crop:
                output_path = self.apply_crop(output_path, segment.metadata.crop)
                
            # Save metadata
            metadata_path = str(self.config.output_dir / f"{Path(output_path).stem}.json")
            with open(metadata_path, "w") as f:
                json.dump(segment.metadata.dict(), f, indent=4)
                
            # Update stats
            self.stats.total_videos += 1
            self.stats.processing_times.append(segment.processing_time.timestamp())
            
            return output_path, metadata_path
            
        except Exception as e:
            logger.error(f"Error saving segment: {str(e)}")
            self.stats.error_count += 1
            raise
            
    def delete_segment(self, segment: VideoSegment) -> None:
        """Mark a segment as deleted"""
        segment.metadata.is_deleted = True
        
    def get_segment_preview(self, video_path: str, time: float) -> np.ndarray:
        """Get a preview frame from the video at specified time"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(time * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise RuntimeError(f"Failed to read frame at time {time}")
                
            return frame
            
        except Exception as e:
            logger.error(f"Error getting preview: {str(e)}")
            raise 