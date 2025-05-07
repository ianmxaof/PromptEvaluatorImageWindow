"""
Core processors for image handling and self-training
"""
import logging
from pathlib import Path
from typing import Callable, Optional, List, Dict, Tuple
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import threading
import torch
import torch.multiprocessing as mp
import psutil
import cv2
import numpy as np
import os
import json
from datetime import datetime
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from .config import CATEGORY_GROUPS
from ..utils.logger import log_action

# Define category prompts
CATEGORY_PROMPTS = {
    # Body part focused categories
    "tits": "large breasts, big tits, prominent chest, cleavage, bare breasts, topless, breasts as main focus, close up of breasts",
    "ass": "large buttocks, big ass, prominent posterior, buttocks as main focus, close up of buttocks, backside view",
    
    # Type based
    "amateur": "amateur, homemade, self-shot, natural, authentic, personal photos, casual setting, home environment, bedroom, bathroom, living room",
    "professional": "professional, studio quality, high production, commercial, polished, studio setting, professional lighting, backdrop, staged",
    
    # Ethnicity based
    "asian": "asian, asian features, asian ethnicity, asian appearance, asian model, asian woman, asian man",
    "european": "european, european features, european ethnicity, european appearance, european model, european woman, european man",
    "american": "american, american features, american ethnicity, american appearance, american model, american woman, american man",
    
    # Special categories
    "lesbian": "lesbian, female same-sex, women together, female couples, two women, female intimacy",
    "gay": "gay, male same-sex, men together, male couples, two men, male intimacy",
    "trans": "transgender, trans features, gender diverse, non-binary, trans woman, trans man",
    
    # Style based
    "fetish": "fetish, kinky, unusual preferences, specific interests, niche, specialized interests",
    "bdsm": "BDSM, bondage, domination, submission, masochism, leather, restraints, chains, whips",
    "cosplay": "cosplay, costume play, character roleplay, dressed up, costumes, character outfit, themed clothing",
    "hentai": "hentai, anime style, japanese animation, cartoon style, animated, drawn style",
    "manga": "manga, japanese comics, drawn style, comic art, comic book style, illustrated",
    "vintage": "vintage, retro style, old school, classic, historical, period piece, nostalgic",
    
    # General
    "other": "other, miscellaneous, not fitting other categories, general, unclear focus"
}

# Special detection prompts
DETECTION_PROMPTS = {
    "emoji_blur": "emoji overlay, emoji sticker, emoji covering, emoji hiding, emoji mask, emoji blurring",
    "face_blur": "blurred face, obscured face, hidden face, face covered, face hidden, face blocked",
    "extreme_crop": "extremely cropped, tight crop, close crop, cropped edges, partial view, cropped image",
    "low_quality": "low quality, pixelated, blurry, grainy, poor resolution, web-scraped image",
    "clear_selfie": "clear selfie, visible face, unobscured selfie, clear face, good quality selfie",
    "environment": "environmental background, scenery, landscape, no people, empty scene"
}

class ImageProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.observer = None
        self.is_watching = False
        self.callback = None
        self.model = None
        self.processor = None
        self.device = None
        self.face_cascade = None
        self.self_trainer = SelfTrainer()
        self._load_model()
        
    def _load_model(self):
        """Load CLIP model and processor"""
        try:
            log_action("Loading CLIP model...")
            
            # Set device (GPU if available, otherwise CPU with limited threads)
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                log_action("Using GPU for processing")
            else:
                self.device = torch.device("cpu")
                # Limit CPU threads
                torch.set_num_threads(2)
                log_action("Using CPU with limited threads (2)")
            
            # Use the model from self-trainer
            self.model = self.self_trainer.model
            self.processor = self.self_trainer.processor
            
            # Load face detection cascade
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            log_action("CLIP model loaded successfully")
        except Exception as e:
            log_action(f"Error loading CLIP model: {str(e)}")
            raise
        
    def start_watching(self, watch_folder: Path, source_folder: Path, callback: Callable):
        """Start watching for new images"""
        if self.is_watching:
            return
            
        self.callback = callback
        self.observer = Observer()
        event_handler = FileSystemEventHandler()
        self.observer.schedule(event_handler, str(watch_folder), recursive=False)
        self.observer.start()
        self.is_watching = True
        
    def stop_watching(self):
        """Stop watching for new images"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            self.is_watching = False
            
    def manual_refresh(self, watch_folder: Path, source_folder: Path, callback: Callable):
        """Manually process files in watch folder"""
        if not watch_folder.exists():
            raise ValueError("Watch folder does not exist")
            
        if not source_folder.exists():
            raise ValueError("Source folder does not exist")
            
        # Simulate processing
        total_files = len(list(watch_folder.glob('*')))
        if total_files == 0:
            callback(1.0)
            return
            
        for i, file_path in enumerate(watch_folder.glob('*')):
            if file_path.is_file():
                # Simulate categorization
                categories = ["anime", "art", "cars", "cats", "dogs", "food", 
                            "landscape", "memes", "nature", "people", "space", 
                            "sports", "technology"]
                import random
                category = random.choice(categories)
                
                # Update progress
                progress = (i + 1) / total_files
                callback(progress, category)
                time.sleep(0.1)  # Simulate processing time

    def process_image(self, image_path: str) -> Tuple[str, Dict]:
        """Process a single image and return its category and metadata"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Get prompts for all categories
            prompts = list(CATEGORY_PROMPTS.values())
            
            # Process image and text
            inputs = self.processor(images=image, text=prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get features
            with torch.no_grad():
                image_features = self.model.get_image_features(inputs["pixel_values"])
                text_features = self.model.get_text_features(inputs["input_ids"])
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            similarity = similarity.cpu()
            
            # Get top category
            top_idx = similarity[0].argmax().item()
            top_category = list(CATEGORY_PROMPTS.keys())[top_idx]
            top_score = float(similarity[0][top_idx])
            
            # Get metadata
            metadata = {
                "category": top_category,
                "confidence": top_score,
                "faces_detected": self._detect_faces(image_path),
                "special_features": self._detect_special_features(image_path)
            }
            
            # Add to training data if confidence is high enough
            if top_score > 0.8:
                self.self_trainer.add_training_example(image_path, top_category, top_score)
                
            return top_category, metadata
            
        except Exception as e:
            log_action(f"Error processing image {image_path}: {str(e)}")
            return "other", {"error": str(e)}
            
    def _detect_faces(self, image_path: str) -> int:
        """Detect number of faces in image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return 0
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            return len(faces)
        except Exception as e:
            log_action(f"Error detecting faces: {str(e)}")
            return 0
            
    def _detect_special_features(self, image_path: str) -> Dict[str, bool]:
        """Detect special features like emoji blur, face blur, etc."""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Get prompts for special features
            prompts = list(DETECTION_PROMPTS.values())
            
            # Process image and text
            inputs = self.processor(images=image, text=prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get features
            with torch.no_grad():
                image_features = self.model.get_image_features(inputs["pixel_values"])
                text_features = self.model.get_text_features(inputs["input_ids"])
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            similarity = similarity.cpu()
            
            # Convert to dictionary
            features = {name: float(score) > 0.25 for name, score in zip(DETECTION_PROMPTS.keys(), similarity[0])}
            
            # Add filename-based heuristics
            features.update({
                "emojis_detected": any(emo in image_path.lower() for emo in ["😈","🍑","emoji"]),
                "blur_detected": "blur" in image_path.lower(),
                "crop_level": "extreme" if "crop" in image_path.lower() else "normal"
            })
            
            return features
            
        except Exception as e:
            log_action(f"Error detecting special features: {str(e)}")
            return {name: False for name in DETECTION_PROMPTS.keys()}

    def provide_feedback(self, image_path: str, correct_category: str):
        """Provide feedback for incorrect categorization"""
        try:
            # Add to training data with high confidence
            self.self_trainer.add_training_example(image_path, correct_category, 1.0)
            
            # Trigger training if we have enough new examples
            if len(self.self_trainer.training_data) >= 10:
                self.self_trainer.train()
                
        except Exception as e:
            log_action(f"Error providing feedback: {str(e)}")
            
    def get_training_stats(self) -> Dict:
        """Get statistics about training data"""
        return self.self_trainer.get_training_stats()

class SelfTrainer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.training_data = []
        self.model = None
        self.processor = None
        self.device = None
        self.training_dir = Path.home() / ".organizerbot" / "training"
        self.model_dir = self.training_dir / "models"
        self._setup_directories()
        self._load_model()
        
    def _setup_directories(self):
        """Setup training directories"""
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_model(self):
        """Load CLIP model and processor"""
        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                log_action("Using GPU for training")
            else:
                self.device = torch.device("cpu")
                torch.set_num_threads(2)
                log_action("Using CPU with limited threads (2) for training")
            
            # Try to load latest fine-tuned model
            latest_model = self._get_latest_model()
            if latest_model:
                log_action(f"Loading fine-tuned model: {latest_model}")
                self.model = CLIPModel.from_pretrained(latest_model).to(self.device)
            else:
                log_action("Loading base CLIP model")
                self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
        except Exception as e:
            log_action(f"Error loading model for training: {str(e)}")
            raise
            
    def _get_latest_model(self) -> Optional[str]:
        """Get the path to the latest fine-tuned model"""
        try:
            models = list(self.model_dir.glob("model_*"))
            if not models:
                return None
            return str(max(models, key=lambda x: x.stat().st_mtime))
        except Exception as e:
            log_action(f"Error finding latest model: {str(e)}")
            return None
            
    def add_training_example(self, image_path: str, category: str, confidence: float = 1.0):
        """Add a training example with confidence score"""
        self.training_data.append({
            "image_path": image_path,
            "category": category,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save training data
        self._save_training_data()
        
    def _save_training_data(self):
        """Save training data to disk"""
        try:
            data_file = self.training_dir / "training_data.json"
            with open(data_file, 'w') as f:
                json.dump(self.training_data, f, indent=2)
        except Exception as e:
            log_action(f"Error saving training data: {str(e)}")
            
    def _load_training_data(self):
        """Load training data from disk"""
        try:
            data_file = self.training_dir / "training_data.json"
            if data_file.exists():
                with open(data_file) as f:
                    self.training_data = json.load(f)
        except Exception as e:
            log_action(f"Error loading training data: {str(e)}")
            
    def train(self, batch_size: int = 32, epochs: int = 3, learning_rate: float = 1e-5):
        """Train the model on collected examples"""
        if not self.training_data:
            log_action("No training data available")
            return
            
        try:
            log_action(f"Starting training on {len(self.training_data)} examples")
            
            # Prepare training data
            images = []
            texts = []
            for example in self.training_data:
                try:
                    image = Image.open(example["image_path"])
                    images.append(image)
                    texts.append(CATEGORY_PROMPTS[example["category"]])
                except Exception as e:
                    log_action(f"Error loading training example: {str(e)}")
                    continue
                    
            if not images:
                log_action("No valid training examples found")
                return
                
            # Prepare optimizer
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
            
            # Training loop
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                batches = 0
                
                # Process in batches
                for i in range(0, len(images), batch_size):
                    batch_images = images[i:i + batch_size]
                    batch_texts = texts[i:i + batch_size]
                    
                    # Process batch
                    inputs = self.processor(
                        images=batch_images,
                        text=batch_texts,
                        return_tensors="pt",
                        padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Forward pass
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batches += 1
                    
                    # Log progress
                    if batches % 10 == 0:
                        avg_loss = total_loss / batches
                        log_action(f"Epoch {epoch + 1}/{epochs}, Batch {batches}, Loss: {avg_loss:.4f}")
                        
            # Save fine-tuned model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.model_dir / f"model_{timestamp}"
            self.model.save_pretrained(model_path)
            log_action(f"Saved fine-tuned model to {model_path}")
            
            # Clean up old models (keep last 3)
            self._cleanup_old_models()
            
        except Exception as e:
            log_action(f"Error during training: {str(e)}")
            
    def _cleanup_old_models(self, keep_last: int = 3):
        """Clean up old model versions, keeping only the most recent ones"""
        try:
            models = sorted(self.model_dir.glob("model_*"), key=lambda x: x.stat().st_mtime)
            for model in models[:-keep_last]:
                model.unlink()
        except Exception as e:
            log_action(f"Error cleaning up old models: {str(e)}")
            
    def get_training_stats(self) -> Dict:
        """Get statistics about training data"""
        try:
            categories = {}
            for example in self.training_data:
                cat = example["category"]
                categories[cat] = categories.get(cat, 0) + 1
                
            return {
                "total_examples": len(self.training_data),
                "categories": categories,
                "latest_model": self._get_latest_model()
            }
        except Exception as e:
            log_action(f"Error getting training stats: {str(e)}")
            return {
                "total_examples": 0,
                "categories": {},
                "latest_model": None
            } 