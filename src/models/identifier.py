import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import logging
import open_clip

from rembg import remove 
from ..utils.color_detector import get_dominant_color 

logger = logging.getLogger(__name__)

class ReferenceEncoder:
    def __init__(self, model_name: str = "convnext_tiny", pretrained: str = "openai"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained,
            device=self.device
        )
        self.model.eval()
        logger.info(f"Model {model_name} loaded on device {self.device}")

    @torch.no_grad()
    def create_target_signature(self, image_paths: list[str]) -> dict:
        """
        Takes reference images, uses 'rembg' to automatically remove
        the background, and creates the multi-modal target signature.
        """
        image_tensors = []
        all_color_features = []

        for path in image_paths:
            try:
                # 1. Load original image
                original_pil_image = Image.open(path)
                clean_pil_image = remove(original_pil_image)

                # 3. Get Color Feature
                mask_cv = np.array(clean_pil_image.getchannel('A'))
                frame_cv = cv2.cvtColor(np.array(original_pil_image), cv2.COLOR_RGB_BGR)
                color_hsv = get_dominant_color(frame_cv, mask_cv)
                
                if color_hsv:
                    all_color_features.append(color_hsv)
                else:
                    logger.warning(f"Could not get color from {path}")
                    continue

                # 4. Preprocess and collect image tensors
                image_tensors.append(self.preprocess(clean_pil_image))

            except Exception as e:
                logger.warning(f"Skipping reference image {path}: {e}")
                continue
        
        if not image_tensors or not all_color_features:
            raise ValueError("Could not create signature. No valid reference images processed.")

        # Stack all preprocessed images into a single batch 
        batch_tensor = torch.stack(image_tensors).to(self.device)
        all_semantic_features = self.model.encode_image(batch_tensor) 

        # Average the batch of features along the 0-th dimension
        master_vector = torch.mean(all_semantic_features, dim=0)
        master_color = tuple(np.mean(all_color_features, axis=0).astype(int))
        
        master_vector = F.normalize(master_vector, p=2, dim=0)
        logger.info(f"Created CLEAN target signature using rembg. Color: {master_color}")
        
        return {
            "semantic_vector": master_vector,
            "color_hsv": master_color
        }

    @torch.no_grad()
    def encode_batch(self, image_crops: list[Image.Image]) -> torch.Tensor:
        """
        Encodes a BATCH of PIL images (from the live video)
        into a stack of feature vectors.
        """
        try:
            # Preprocess all images in the list
            image_tensors = [self.preprocess(img) for img in image_crops]
            batch_tensor = torch.stack(image_tensors).to(self.device)
            embeddings = self.model.encode_image(batch_tensor)
            
            # Normalize and return
            return F.normalize(embeddings, p=2, dim=-1)
            
        except Exception as e:
            logger.error(f"Error in encode_batch: {e}")
            return torch.empty(0) # Return an empty tensor on failure

    def extract_features_batch(self, frame_bgr: np.ndarray, detections: list) -> tuple:
        """
        Extracts semantic and color features from a batch of detections.
        Returns (sem_feats, color_feats) where each is a list.
        """
        sem_feats = []
        color_feats = []
        
        try:
            # Collect all crops as PIL images
            crops = []
            for det in detections:
                if det.crop is not None:
                    crops.append(det.crop)
                else:
                    crops.append(Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)))
            
            if not crops:
                return [], []
            
            # Encode all crops in batch
            embeddings = self.encode_batch(crops)
            
            # Extract color features from masks
            for i, det in enumerate(detections):
                # Semantic feature (from embeddings)
                if i < len(embeddings):
                    sem_feat = embeddings[i].cpu().numpy()
                else:
                    sem_feat = None
                sem_feats.append(sem_feat)
                
                # Color feature from mask
                color_feat = get_dominant_color(frame_bgr, det.mask)
                color_feats.append(color_feat)
            
            return sem_feats, color_feats
            
        except Exception as e:
            logger.error(f"Error in extract_features_batch: {e}")
            return [None] * len(detections), [None] * len(detections)