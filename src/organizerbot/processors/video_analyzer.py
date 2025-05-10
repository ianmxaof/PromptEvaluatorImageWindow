"""
Video analysis module using CLIP for content classification
"""
import os
import subprocess
import torch
import clip
from PIL import Image
import json
from pathlib import Path
import urllib.parse
from typing import List, Dict, Tuple, Optional
import logging
import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    BASIC = "basic"
    SCENE_DETECTION = "scene"
    MOTION_ANALYSIS = "motion"
    FULL = "full"

@dataclass
class SceneInfo:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    dominant_category: str
    confidence: float
    motion_score: float
    tags: Dict[str, float]

class VideoAnalyzer:
    def __init__(self, model_name: str = "ViT-B/32"):
        """Initialize the video analyzer with CLIP model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            logger.info(f"Loaded CLIP model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}")
            raise
            
        # Default categories for classification
        self.categories = [
            "porn", "nature", "documentary", "sports", "anime", "cartoon", 
            "talking", "weapons", "celebrity", "vlog", "gaming", "technology", 
            "nudity", "safe content"
        ]
        
        # Create output directories
        self.output_dir = Path.home() / ".organizerbot" / "output"
        self.frames_dir = self.output_dir / "frames"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
    def _normalize_path(self, path: str) -> Path:
        """Normalize file path to handle special characters"""
        # Decode URL-encoded path
        decoded_path = urllib.parse.unquote(path)
        # Convert to absolute path
        return Path(decoded_path).resolve()
        
    def extract_frames(self, video_path: str, interval: int = 60) -> List[str]:
        """Extract frames from video at specified intervals"""
        try:
            video_path = self._normalize_path(video_path)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
                
            # Create a unique directory for this video's frames
            video_frames_dir = self.frames_dir / video_path.stem
            video_frames_dir.mkdir(parents=True, exist_ok=True)
            
            output_pattern = str(video_frames_dir / 'frame_%03d.jpg')
            # Use a more reliable FFmpeg command with proper encoding parameters
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vf', f'fps=1/{interval}',
                '-q:v', '2',  # High quality JPEG
                '-f', 'image2',
                '-vcodec', 'mjpeg',
                '-y',  # Overwrite output files
                output_pattern
            ]
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr.decode()}")
                raise RuntimeError("Failed to extract frames")
                
            return sorted([str(p) for p in video_frames_dir.glob('*.jpg')])
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise
            
    def tag_image(self, image_path: str) -> Dict[str, float]:
        """Tag a single image using CLIP"""
        try:
            image_path = self._normalize_path(image_path)
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            text_inputs = clip.tokenize(self.categories).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(text_inputs)
                logits_per_image = image_features @ text_features.T
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
                
            # Convert numpy float32 values to Python floats
            return dict(sorted([(cat, float(prob)) for cat, prob in zip(self.categories, probs)], key=lambda x: -x[1])[:5])
            
        except Exception as e:
            logger.error(f"Error tagging image {image_path}: {str(e)}")
            raise
            
    def detect_scenes(self, video_path: str, threshold: float = 30.0) -> List[SceneInfo]:
        """Detect scene changes in video"""
        try:
            video_path = self._normalize_path(video_path)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
                
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            scenes = []
            prev_frame = None
            scene_start = 0
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(gray, prev_frame)
                    score = np.mean(diff)
                    
                    # Detect scene change
                    if score > threshold:
                        # Get frame tags
                        frame_path = self.frames_dir / f"temp_frame_{frame_count}.jpg"
                        cv2.imwrite(str(frame_path), frame)
                        tags = self.tag_image(str(frame_path))
                        frame_path.unlink()  # Delete temp file
                        
                        # Create scene info
                        scene = SceneInfo(
                            start_frame=scene_start,
                            end_frame=frame_count,
                            start_time=scene_start / fps,
                            end_time=frame_count / fps,
                            dominant_category=max(tags.items(), key=lambda x: x[1])[0],
                            confidence=max(tags.values()),
                            motion_score=score,
                            tags=tags
                        )
                        scenes.append(scene)
                        scene_start = frame_count
                
                prev_frame = gray
                frame_count += 1
                
            cap.release()
            return scenes
            
        except Exception as e:
            logger.error(f"Error detecting scenes: {str(e)}")
            raise
            
    def analyze_motion(self, video_path: str, interval: int = 60) -> List[Dict[str, float]]:
        """Analyze motion in video frames"""
        try:
            video_path = self._normalize_path(video_path)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
                
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            motion_scores = []
            prev_frame = None
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % interval == 0:
                    # Convert to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    if prev_frame is not None:
                        # Calculate optical flow
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                        )
                        
                        # Calculate motion magnitude
                        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                        motion_score = np.mean(magnitude)
                        
                        motion_scores.append({
                            "frame": frame_count,
                            "time": frame_count / fps,
                            "motion_score": float(motion_score)
                        })
                    
                    prev_frame = gray
                
                frame_count += 1
                
            cap.release()
            return motion_scores
            
        except Exception as e:
            logger.error(f"Error analyzing motion: {str(e)}")
            raise
            
    def create_metadata_manifest(self, frame_paths: List[str], output_path: str = "video_manifest.json") -> str:
        """Create metadata manifest for video frames"""
        try:
            manifest = []
            for i, path in enumerate(frame_paths):
                tags = self.tag_image(path)
                # Convert float32 values to regular Python floats
                tags = {k: float(v) for k, v in tags.items()}
                manifest.append({
                    "frame": os.path.basename(path),
                    "timestamp": f"{i * 60}s",
                    "tags": tags
                })
                
            output_path = self.output_dir / output_path
            with open(output_path, "w") as f:
                json.dump(manifest, f, indent=4)
                
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating metadata manifest: {str(e)}")
            raise
            
    def make_grid_thumbnail(self, image_paths: List[str], grid_width: int = 5, output_path: str = "preview_grid.jpg") -> str:
        """Create a grid thumbnail from frame images"""
        try:
            images = [Image.open(self._normalize_path(p)) for p in image_paths]
            w, h = images[0].size
            grid_height = (len(images) + grid_width - 1) // grid_width
            
            grid_img = Image.new('RGB', (w * grid_width, h * grid_height))
            for idx, img in enumerate(images):
                x = (idx % grid_width) * w
                y = (idx // grid_width) * h
                grid_img.paste(img, (x, y))
                
            output_path = self.output_dir / output_path
            grid_img.save(output_path)
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating grid thumbnail: {str(e)}")
            raise
            
    def analyze_video(self, video_path: str, interval: int = 60, show_grid: bool = True, 
                     analysis_type: AnalysisType = AnalysisType.BASIC) -> Tuple[str, Optional[str]]:
        """Analyze video file and generate metadata and preview"""
        try:
            video_path = self._normalize_path(video_path)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
                
            frames = self.extract_frames(str(video_path), interval=interval)
            manifest_path = self.create_metadata_manifest(frames)
            grid_path = self.make_grid_thumbnail(frames) if show_grid else None
            
            # Additional analysis based on type
            if analysis_type in [AnalysisType.SCENE_DETECTION, AnalysisType.FULL]:
                scenes = self.detect_scenes(str(video_path))
                # Save scene data
                scene_path = self.output_dir / f"{video_path.stem}_scenes.json"
                with open(scene_path, "w") as f:
                    json.dump([vars(s) for s in scenes], f, indent=4)
                    
            if analysis_type in [AnalysisType.MOTION_ANALYSIS, AnalysisType.FULL]:
                motion_data = self.analyze_motion(str(video_path), interval)
                # Save motion data
                motion_path = self.output_dir / f"{video_path.stem}_motion.json"
                with open(motion_path, "w") as f:
                    json.dump(motion_data, f, indent=4)
                    
            return manifest_path, grid_path
            
        except Exception as e:
            logger.error(f"Error analyzing video {video_path}: {str(e)}")
            raise
            
    def cleanup(self, frame_dir: str = 'frames'):
        """Clean up temporary files"""
        try:
            frame_dir = Path(frame_dir)
            if frame_dir.exists():
                for file in frame_dir.glob('*.jpg'):
                    file.unlink()
                frame_dir.rmdir()
        except Exception as e:
            logger.error(f"Error cleaning up: {str(e)}")
            raise 