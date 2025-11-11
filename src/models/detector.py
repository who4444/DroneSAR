import logging
import numpy as np
from PIL import Image
import torch
from sahi.slicing import slice_image
import cv2
from ultralytics import YOLO
from ..utils.color_detector import Detection


logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(self, checkpoint_path: str, tiling_config: dict):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Save tiling configuration
        self.tiling_config = tiling_config

        self.checkpoint = checkpoint_path
        self.model = YOLO(self.checkpoint)
        self.model.to(self.device)  # Move the model to the device
        
    def _parse_yolo_results(self, results, full_shape_hw, offset_x, offset_y) -> list[Detection]:
        """
        Parses the 'ultralytics' Results object for segmentation.
        """
        parsed_detections = []
        
        # 'results' is a list (one item per image). Get the first one.
        if not results or not results[0]:
            return []

        image_results = results[0]
        
        # Get the shape of the tile (the original image for this prediction)
        tile_h, tile_w = image_results.orig_shape

        # Check if any masks were detected
        if image_results.masks is None:
            return []

        # Iterate through all detected objects (boxes and masks are parallel)
        for box, mask in zip(image_results.boxes, image_results.masks):
            
            # --- 1. Get Confidence ---
            conf = box.conf[0].item()

            # --- 2. Get Bounding Box (relative to the tile) ---
            bbox_tile_xyxy = box.xyxy[0].cpu().numpy()
            x1_tile, y1_tile, x2_tile, y2_tile = bbox_tile_xyxy.astype(int)

            # --- 3. Calculate Full-Frame Bounding Box ---
            x1_full = x1_tile + offset_x
            y1_full = y1_tile + offset_y
            x2_full = x2_tile + offset_x
            y2_full = y2_tile + offset_y
            bbox_xyxy_full = np.array([x1_full, y1_full, x2_full, y2_full])

            # --- 4. Get Mask ---
            # 'mask.data' is a (H_proto, W_proto) float tensor (e.g., 160x160)
            mask_data = mask.data[0].cpu().numpy() 
            
            # --- 5. Resize mask prototype to the TILE's size ---
            # We use INTER_LINEAR for resizing floats, then threshold.
            mask_tile_float = cv2.resize(mask_data, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
            
            # --- 6. Threshold to get a binary mask (size of the tile) ---
            mask_tile_binary = (mask_tile_float > 0.5).astype(np.uint8)

            # --- 7. Create a full-frame mask ---
            # This 'full_mask' will contain ONLY this one object's mask
            full_mask = np.zeros(full_shape_hw, dtype=np.uint8)
            
            # Calculate the "paste" coordinates
            y_paste_start = offset_y
            y_paste_end = offset_y + tile_h
            x_paste_start = offset_x
            x_paste_end = offset_x + tile_w
            
            # Paste the binary tile mask onto the full-frame canvas
            # This is the simplest way to reconstruct. NMS will handle overlaps.
            full_mask[y_paste_start:y_paste_end, x_paste_start:x_paste_end] = mask_tile_binary

            parsed_detections.append(
                Detection(
                    bbox=bbox_xyxy_full,
                    mask=full_mask,  # This mask is on the full-frame
                    confidence=conf
                )
            )
            
        return parsed_detections

    def find_objects_with_tiling(self, frame_bgr: np.ndarray) -> list[Detection]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        full_shape_hw = frame_rgb.shape[:2]
    
        # 1. Slice the image
        slice_result = slice_image(
            image=frame_rgb,
            slice_height=self.tiling_config["slice_height"],
            slice_width=self.tiling_config["slice_width"],
            overlap_height_ratio=self.tiling_config["overlap_height_ratio"],
            overlap_width_ratio=self.tiling_config["overlap_width_ratio"],
        )
        
        all_detections = []
        
        # 2. Get the actual list of slices from the slicing result
        for slice_data in slice_result:
            tile = slice_data['image'] # This is a numpy array
            offset_x, offset_y = slice_data['starting_pixel']
            
            # Run prediction on the tile (this is the RGB numpy array)
            yolo_results = self.model.predict(tile, verbose=False)
            
            # Parse the results and map them to full-frame coordinates
            tile_detections = self._parse_yolo_results(
                yolo_results, full_shape_hw, offset_x, offset_y
            )
            all_detections.extend(tile_detections)
            
        # TODO: Add Non-Maximal Suppression (NMS) here
        # You MUST use mask NMS or bbox NMS to merge duplicate
        # detections from the overlapping tiles.
        # Without this, you will have many duplicates.
        
        final_detections = all_detections # Assumes NMS is done (or not needed yet)
        
        # Add the 'crop' to the final detections
        for det in final_detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            # Clip coordinates to be within the image bounds
            y1, y2 = max(0, y1), min(full_shape_hw[0], y2)
            x1, x2 = max(0, x1), min(full_shape_hw[1], x2)
            
            # Crop from the original RGB frame
            det.crop = Image.fromarray(frame_rgb[y1:y2, x1:x2])
            
        return final_detections