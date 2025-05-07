"""
Self-training functionality for OrganizerBot
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import torch
from typing import List, Dict, Optional
from PIL import Image
from organizerbot.utils.logger import log_action

class SelfTrainer:
    def __init__(self, log_path: str = "misclass_log.json"):
        self.log_path = log_path
        self.data = self._load_log()
        self.prompts = None
        self.classifier = None
        self._model = None
        self._processor = None
        self._device = None

    def _load_log(self) -> List[Dict]:
        """Load misclassification log"""
        try:
            with open(self.log_path) as f:
                return json.load(f)
        except Exception as e:
            log_action(f"Error loading log: {str(e)}")
            return []

    def _load_model(self):
        """Lazily load CLIP model"""
        if self._model is None or self._processor is None:
            try:
                from transformers import CLIPProcessor, CLIPModel
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self._device)
                self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            except Exception as e:
                log_action(f"Error loading model: {str(e)}")
                raise

    def embed_image(self, image_path: str) -> np.ndarray:
        """Get CLIP embedding for an image"""
        try:
            self._load_model()
            img = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=img, return_tensors="pt", padding=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                features = self._model.get_image_features(inputs["pixel_values"])
                features /= features.norm(dim=-1, keepdim=True)
            
            return features.cpu().numpy().squeeze(0)
        except Exception as e:
            log_action(f"Error embedding image: {str(e)}")
            return np.zeros(512)  # Return zero vector on error

    def train_classifier(self):
        """Train classifier on labeled data"""
        try:
            # Filter entries with human-corrected labels
            labeled = [e for e in self.data if e.get("correct_label")]
            if not labeled:
                log_action("No labeled data to train on.")
                return

            # Get embeddings and labels
            X = np.array([self.embed_image(e["image_path"]) for e in labeled])
            y = np.array([e["correct_label"] for e in labeled])

            # Train classifier
            self.classifier = LogisticRegression(max_iter=200)
            self.classifier.fit(X, y)
            log_action(f"Classifier trained on {len(y)} samples")
            
            # Save classifier
            self.save_classifier()
            
        except Exception as e:
            log_action(f"Error training classifier: {str(e)}")

    def suggest_prompts(self, n_clusters: int = 5) -> List[str]:
        """Suggest new prompts based on misclassified images"""
        try:
            if not self.data:
                return []

            # Get embeddings for all misclassified images
            X = np.array([self.embed_image(e["image_path"]) for e in self.data])
            
            # Cluster embeddings
            kmeans = KMeans(n_clusters=n_clusters).fit(X)
            centers = kmeans.cluster_centers_
            
            # Convert centers to text prompts
            prompts = []
            for i, center in enumerate(centers):
                # Find closest image in cluster
                distances = np.linalg.norm(X - center, axis=1)
                closest_idx = np.argmin(distances)
                closest_image = self.data[closest_idx]
                
                # Create prompt based on image metadata
                prompt = f"Cluster {i+1}: "
                if closest_image.get("faces_detected", 0) > 0:
                    prompt += "face visible, "
                if closest_image.get("emojis_detected", False):
                    prompt += "emoji present, "
                if closest_image.get("blur_detected", False):
                    prompt += "blurred, "
                if closest_image.get("crop_level") == "extreme":
                    prompt += "tightly cropped, "
                prompt += closest_image.get("top_prompt", "unknown features")
                
                prompts.append(prompt)
            
            return prompts
            
        except Exception as e:
            log_action(f"Error suggesting prompts: {str(e)}")
            return []

    def save_classifier(self, path: str = "classifier.pkl"):
        """Save trained classifier"""
        try:
            import joblib
            if self.classifier:
                joblib.dump(self.classifier, path)
                log_action(f"Saved classifier to {path}")
        except Exception as e:
            log_action(f"Error saving classifier: {str(e)}")

    def load_classifier(self, path: str = "classifier.pkl"):
        """Load trained classifier"""
        try:
            import joblib
            self.classifier = joblib.load(path)
            log_action(f"Loaded classifier from {path}")
        except Exception as e:
            log_action(f"Error loading classifier: {str(e)}")

    def predict(self, image_path: str) -> Optional[str]:
        """Predict category using trained classifier"""
        try:
            if not self.classifier:
                return None
            
            # Get image embedding
            embedding = self.embed_image(image_path)
            
            # Predict category
            prediction = self.classifier.predict([embedding])[0]
            confidence = self.classifier.predict_proba([embedding]).max()
            
            if confidence > 0.5:  # Only return predictions with high confidence
                return prediction
            return None
            
        except Exception as e:
            log_action(f"Error predicting category: {str(e)}")
            return None 