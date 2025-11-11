import logging
import cv2
import numpy as np
from tqdm import tqdm
import torch
import os
import sys

# Ensure models can be imported
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from models.detector import YOLODetector
from models.identifier import ReferenceEncoder
from models.tracker import ObjectTracker
from utils.color_detector import color_distance, draw_tracks

logger = logging.getLogger(__name__)

class SearchPipeline:
    def __init__(self, config):
        self.config = config
        
        logger.info("Initializing pipeline components...")
        self.finder = YOLODetector(
            checkpoint_path=config['finder']['checkpoint'],
            tiling_config=config['finder']['tiling']
        )
        self.identifier = ReferenceEncoder(
            model_name=config['identifier']['model_name']
        )
        self.tracker = ObjectTracker(config['tracker'])
        
        self.sem_threshold = config['identification']['semantic_threshold']
        self.color_threshold = config['identification']['color_threshold']
        
        logger.info("Creating 2-factor target signature...")
        self.target_signature = self.identifier.create_target_signature(
            config['reference_images']
        )
        logger.info("Pipeline initialized successfully.")

    def _is_target(self, sem_feat, color_feat) -> bool:
        """Performs the 2-factor identification check."""
        if sem_feat is None or color_feat is None:
            return False

        sem_sim = np.dot(sem_feat, self.target_signature["semantic_vector"])
        color_dist = color_distance(color_feat, self.target_signature["color_hsv"])
        
        is_sem_match = sem_sim >= self.sem_threshold
        is_color_match = color_dist <= self.color_threshold
        
        return is_sem_match and is_color_match

    def _process_frame(self, frame_bgr, frame_number): # MODIFIED
        """Processes a single BGR frame and updates the tracker."""
        
        # 1. FIND (with Tiling)
        detections = self.finder.find_objects_with_tiling(frame_bgr)
        if not detections:
            return # No detections, just return

        # 2. EXTRACT (2-Factor Features)
        sem_feats, color_feats = \
            self.identifier.extract_features_batch(frame_bgr, detections)

        # 3. IDENTIFY (Logic)
        is_target_flags = [
            self._is_target(s, c) 
            for s, c in zip(sem_feats, color_feats)
        ]
        
        # 4. TRACKER (MODIFIED)
        # This updates self.tracker.tracks internally
        self.tracker.update(
            detections, 
            sem_feats, 
            color_feats, 
            is_target_flags,
            frame_number # Pass frame number
        )

    def _format_submission(self, video_id: str) -> dict:
        """
        Formats the final output for this video according to the schema.
        """
        video_result = {
            "video_id": video_id,
            "detections": []
        }
        
        # Iterate over all tracks that existed during the video
        for track in self.tracker.tracks:
            # Only include tracks that were stable (min_hits) AND flagged as a target
            if track.is_target and track.hits >= self.tracker.min_hits:
                
                # The schema wants a "bboxes" list for each detected object
                video_result["detections"].append({
                    "bboxes": track.bbox_history
                })
        
        return video_result

    def run_on_video(self) -> dict: # MODIFIED: Returns a dict
        """
        Main public method to run the pipeline on a single video.
        Returns a dictionary formatted for the submission.
        """
        
        in_path = self.config['video_input_path']
        cap = cv2.VideoCapture(in_path)
        
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {in_path}")
            # Return empty result per submission requirement
            return {"video_id": os.path.basename(in_path), "detections": []}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {in_path}")
        pbar = tqdm(total=total_frames, desc=f"Processing {os.path.basename(in_path)}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Process the frame (this updates self.tracker)
            self._process_frame(frame, frame_number)
            pbar.update(1)

        cap.release()
        pbar.close()
        
        # Get video_id from the path
        video_id = os.path.basename(in_path)
        
        # Format the results for this video
        video_submission = self._format_submission(video_id)
        
        logger.info(f"Processing complete for {video_id}.")
        return video_submission