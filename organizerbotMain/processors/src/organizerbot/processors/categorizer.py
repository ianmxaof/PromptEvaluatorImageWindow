"""
AI-powered image categorization for OrganizerBot
"""
import os
from typing import List, Dict, Optional
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from organizerbot.core.config import load_config
from organizerbot.utils.logger import log_action

class ImageCategorizer:
    """AI-powered image categorization using CLIP"""
    def __init__(self):
        self.config = load_config()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Load custom categories from config
        self.categories = self.config.get('categories', [
            "nature", "people", "animals", "food", "architecture",
            "art", "technology", "sports", "vehicles", "other"
        ])

    def get_category_scores(self, image: Image.Image) -> Dict[str, float]:
        """
        Get similarity scores for each category
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary of category scores
        """
        # Prepare inputs
        inputs = self.processor(
            text=self.categories,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).squeeze(0)
        
        # Convert to dictionary
        return {cat: float(prob) for cat, prob in zip(self.categories, probs)}

    def categorize_image(self, image_path: str) -> str:
        """
        Categorize an image and return the best matching category
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Best matching category name
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            
            # Get category scores
            scores = self.get_category_scores(image)
            
            # Find best matching category
            best_category = max(scores.items(), key=lambda x: x[1])[0]
            confidence = scores[best_category]
            
            log_action(f"Image categorized as '{best_category}' with confidence {confidence:.2f}")
            return best_category
            
        except Exception as e:
            log_action(f"Error categorizing image: {str(e)}")
            return "other"

    def suggest_categories(self, image_path: str, top_k: int = 3) -> List[str]:
        """
        Get top-k suggested categories for an image
        
        Args:
            image_path: Path to the image file
            top_k: Number of top categories to return
            
        Returns:
            List of top-k categories
        """
        try:
            image = Image.open(image_path)
            scores = self.get_category_scores(image)
            
            # Sort categories by score and get top-k
            top_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return [cat for cat, _ in top_categories]
            
        except Exception as e:
            log_action(f"Error suggesting categories: {str(e)}")
            return ["other"]

def categorize_image(image_path: str) -> str:
    """Categorize an image using the default categorizer"""
    categorizer = ImageCategorizer()
    return categorizer.categorize_image(image_path)

def suggest_categories(image_path: str, top_k: int = 3) -> List[str]:
    """Get suggested categories using the default categorizer"""
    categorizer = ImageCategorizer()
    return categorizer.suggest_categories(image_path, top_k) 