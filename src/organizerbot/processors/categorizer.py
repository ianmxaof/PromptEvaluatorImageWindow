"""
Image categorization functionality for OrganizerBot
"""
import torch
import torch.multiprocessing as mp
import time
import psutil
import cv2
import numpy as np
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from organizerbot.utils.logger import log_action

# Define categories with detailed prompts
CATEGORY_PROMPTS = {
    # Body part focused categories
    "tits": "large breasts, big tits, prominent chest, cleavage, bare breasts, topless, breasts as main focus, close up of breasts",
    "ass": "large buttocks, big ass, prominent posterior, buttocks as main focus, close up of buttocks, backside view",
    
    # Selfie specific
    "selfie": "selfie, phone camera, mirror selfie, front camera, self portrait, holding phone, phone visible, self shot, arm extended",
    
    # Original categories with enhanced prompts
    "amateur": "amateur, homemade, self-shot, natural, authentic, personal photos, casual setting, home environment, bedroom, bathroom, living room",
    "professional": "professional, studio quality, high production, commercial, polished, studio setting, professional lighting, backdrop, staged",
    "asian": "asian, asian features, asian ethnicity, asian appearance, asian model, asian woman, asian man",
    "european": "european, european features, european ethnicity, european appearance, european model, european woman, european man",
    "american": "american, american features, american ethnicity, american appearance, american model, american woman, american man",
    "lesbian": "lesbian, female same-sex, women together, female couples, two women, female intimacy",
    "gay": "gay, male same-sex, men together, male couples, two men, male intimacy",
    "trans": "transgender, trans features, gender diverse, non-binary, trans woman, trans man",
    "fetish": "fetish, kinky, unusual preferences, specific interests, niche, specialized interests",
    "bdsm": "BDSM, bondage, domination, submission, masochism, leather, restraints, chains, whips",
    "cosplay": "cosplay, costume play, character roleplay, dressed up, costumes, character outfit, themed clothing",
    "hentai": "hentai, anime style, japanese animation, cartoon style, animated, drawn style",
    "manga": "manga, japanese comics, drawn style, comic art, comic book style, illustrated",
    "vintage": "vintage, retro style, old school, classic, historical, period piece, nostalgic",
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

# CPU optimization settings
MAX_CPU_PERCENT = 60
BATCH_INTERVAL = 5  # seconds
MAX_WORKERS = 2
MAX_QUEUE_SIZE = 10
THROTTLE_SLEEP = 0.2  # seconds

# Global variables
_model = None
_processor = None
_device = None
_face_cascade = None
_classifier = None

# Processing queue and executor
inference_queue = Queue(maxsize=MAX_QUEUE_SIZE)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Logging
LOG_PATH = os.getenv("MISCLASS_LOG_PATH", "misclass_log.json")

def log_misclassified(entry: Dict):
    """Log misclassified images for training"""
    logs = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r') as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    logs.append(entry)
    with open(LOG_PATH, 'w') as f:
        json.dump(logs, f, indent=2)

def _load_model():
    """Lazily load the CLIP model and processor"""
    global _model, _processor, _device, _face_cascade
    if _model is None or _processor is None:
        try:
            log_action("Loading CLIP model...")
            
            # Set device (GPU if available, otherwise CPU with limited threads)
            if torch.cuda.is_available():
                _device = torch.device("cuda")
                log_action("Using GPU for processing")
            else:
                _device = torch.device("cpu")
                # Limit CPU threads
                torch.set_num_threads(2)
                log_action("Using CPU with limited threads (2)")
            
            _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device)
            _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Load face detection cascade
            _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Test the model
            test_image = Image.new('RGB', (224, 224), color='red')
            test_text = ["test"]
            test_inputs = _processor(images=test_image, text=test_text, return_tensors="pt", padding=True)
            test_inputs = {k: v.to(_device) for k, v in test_inputs.items()}
            
            with torch.no_grad():
                _model.get_image_features(test_inputs["pixel_values"])
                _model.get_text_features(test_inputs["input_ids"])
            
            log_action("CLIP model loaded and tested successfully")
        except Exception as e:
            log_action(f"Error loading CLIP model: {str(e)}")
            raise

def detect_faces(image_path: str) -> int:
    """Detect number of faces in image"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return 0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade.detectMultiScale(gray, 1.3, 5)
        return len(faces)
    except Exception as e:
        log_action(f"Error detecting faces: {str(e)}")
        return 0

def detect_special_features(image_path: str) -> Dict[str, bool]:
    """Detect special features like emoji blur, face blur, etc."""
    try:
        # Load image
        image = Image.open(image_path)
        
        # Get prompts for special features
        prompts = list(DETECTION_PROMPTS.values())
        
        # Process image and text
        inputs = _processor(images=image, text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        
        # Get features
        with torch.no_grad():
            image_features = _model.get_image_features(inputs["pixel_values"])
            text_features = _model.get_text_features(inputs["input_ids"])
        
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

def safe_run(func, *args, **kwargs):
    """Run function only if CPU usage is below threshold"""
    while psutil.cpu_percent(interval=1) > MAX_CPU_PERCENT:
        log_action("CPU usage high, waiting...")
        time.sleep(2)
    return func(*args, **kwargs)

def process_batch(image_paths: List[str]) -> Dict[str, Tuple[str, Dict]]:
    """Process a batch of images"""
    results = {}
    try:
        # Ensure model is loaded
        _load_model()
        
        # Process all images in batch
        images = [Image.open(path) for path in image_paths]
        prompts = list(CATEGORY_PROMPTS.values())
        
        # Process images and text
        inputs = _processor(images=images, text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        
        # Get features
        with torch.no_grad():
            image_features = _model.get_image_features(inputs["pixel_values"])
            text_features = _model.get_text_features(inputs["input_ids"])
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        # Calculate similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        similarity = similarity.cpu()
        
        # Process results for each image
        for i, path in enumerate(image_paths):
            # Get category scores
            scores = {cat: float(score) for cat, score in zip(CATEGORY_PROMPTS.keys(), similarity[i])}
            best_category = max(scores.items(), key=lambda x: x[1])[0]
            best_score = scores[best_category]
            
            # Get metadata
            faces = detect_faces(path)
            features = detect_special_features(path)
            
            metadata = {
                "image_path": path,
                "timestamp": datetime.utcnow().isoformat(),
                "faces_detected": faces,
                "clip_confidence": best_score,
                "top_prompt": CATEGORY_PROMPTS[best_category],
                **features
            }
            
            # Determine final category
            if best_score < 3.0:
                category = "other"
                # Log misclassified for training
                log_misclassified(metadata)
            else:
                category = best_category
            
            results[path] = (category, metadata)
            
            # Log results
            log_action(f"Results for {path}:")
            log_action(f"  Category: {category} (confidence: {best_score:.2f}%)")
            log_action(f"  Faces detected: {faces}")
            for feature, detected in features.items():
                log_action(f"  {feature}: {detected}")
        
        return results
        
    except Exception as e:
        log_action(f"Error processing batch: {str(e)}")
        return {path: ("other", {
            "image_path": path,
            "timestamp": datetime.utcnow().isoformat(),
            "faces_detected": 0,
            "clip_confidence": 0.0,
            "top_prompt": "error",
            **{k: False for k in DETECTION_PROMPTS.keys()}
        }) for path in image_paths}

def inference_worker():
    """Worker thread for processing queued images"""
    batch = []
    last_process_time = time.time()
    
    while True:
        try:
            # Get file from queue with timeout
            filepath = inference_queue.get(timeout=1)
            batch.append(filepath)
            
            # Process batch if interval elapsed or queue empty
            current_time = time.time()
            if current_time - last_process_time >= BATCH_INTERVAL or inference_queue.empty():
                if batch:
                    safe_run(process_batch, batch)
                    batch = []
                    last_process_time = current_time
            
            inference_queue.task_done()
            time.sleep(THROTTLE_SLEEP)  # Throttle CPU usage
            
        except Queue.Empty:
            # Process any remaining files in batch
            if batch:
                safe_run(process_batch, batch)
                batch = []
                last_process_time = time.time()
            time.sleep(THROTTLE_SLEEP)
        except Exception as e:
            log_action(f"Error in inference worker: {str(e)}")
            time.sleep(THROTTLE_SLEEP)

def categorize_image(image_path: str) -> Tuple[str, Dict]:
    """
    Categorize an image and return category with metadata
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (category, metadata)
    """
    try:
        # Try to enqueue for batch processing
        try:
            inference_queue.put(image_path, timeout=1)
            return "processing", {"status": "queued"}
        except Queue.Full:
            # If queue is full, process immediately
            results = safe_run(process_batch, [image_path])
            return results[image_path]
            
    except Exception as e:
        log_action(f"Error categorizing image: {str(e)}")
        return "other", {
            "image_path": image_path,
            "timestamp": datetime.utcnow().isoformat(),
            "faces_detected": 0,
            "clip_confidence": 0.0,
            "top_prompt": "error",
            **{k: False for k in DETECTION_PROMPTS.keys()}
        }

def suggest_categories(image_path: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Get top-k suggested categories for an image with confidence scores
    
    Args:
        image_path: Path to the image file
        top_k: Number of top categories to return
        
    Returns:
        List of (category, confidence) tuples
    """
    try:
        # Process image
        results = safe_run(process_batch, [image_path])
        category, metadata = results[image_path]
        
        # Get scores for all categories
        scores = {cat: float(score) for cat, score in zip(CATEGORY_PROMPTS.keys(), metadata["clip_confidence"])}
        
        # Get top-k categories with scores above threshold
        threshold = 1.5  # Lowered threshold to 1.5%
        top_categories = [(cat, score) for cat, score in scores.items() if score > threshold]
        top_categories.sort(key=lambda x: x[1], reverse=True)
        top_categories = top_categories[:top_k]
        
        log_action(f"Top {top_k} categories:")
        for cat, score in top_categories:
            log_action(f"  {cat}: {score:.2f}%")
            log_action(f"    Prompt: {CATEGORY_PROMPTS[cat]}")
        
        return top_categories
        
    except Exception as e:
        log_action(f"Error suggesting categories: {str(e)}")
        return [("other", 0.0)]

# Start worker threads
for _ in range(MAX_WORKERS):
    executor.submit(inference_worker) 