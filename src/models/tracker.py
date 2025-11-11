import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import logging

from ..utils.color_detector import color_distance

logger = logging.getLogger(__name__)

def calculate_color_distance(hsv1, hsv2):
    """Wrapper for color_distance from utils."""
    return color_distance(hsv1, hsv2)

def iou_cost(bbox1, bbox2):
    x1_inter = max(bbox1[0], bbox2[0]); y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2]); y2_inter = min(bbox1[3], bbox2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + 1e-6)
    return 1.0 - iou

def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.; y = bbox[1] + h / 2.
    a = w * h; r = w / (h + 1e-6)
    return np.array([x, y, a, r]).T

def convert_x_to_bbox(x):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / (w + 1e-6)
    return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2.]).T
# ---

class Track:
    _next_id = 1
    
    def __init__(self, detection, sem_feat, color_feat, is_target_flag, min_hits, frame_number): # MODIFIED
        self.track_id = Track._next_id
        Track._next_id += 1
        
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # ... (KF setup logic is UNCHANGED) ...
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.; self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01; self.kf.Q[4:, 4:] *= 0.01
        
        # --- MODIFIED ---
        self.kf.x[:4] = convert_bbox_to_z(detection.bbox)
        self.bbox = detection.bbox # Set initial bbox
        self.semantic_feature = sem_feat
        self.color_hsv = color_feat
        self.is_target = is_target_flag
        
        self.age = 0
        self.hits = 1
        self.min_hits = min_hits
        self.time_since_update = 0

        # --- NEW: Store the first bounding box ---
        self.bbox_history = [{
            "frame": frame_number,
            "x1": int(self.bbox[0]),
            "y1": int(self.bbox[1]),
            "x2": int(self.bbox[2]),
            "y2": int(self.bbox[3])
        }]
        # --- END NEW ---

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        self.bbox = convert_x_to_bbox(self.kf.x)
        return self.bbox

    def update(self, detection, sem_feat, color_feat, is_target_flag, frame_number): # MODIFIED
        self.kf.update(convert_bbox_to_z(detection.bbox))
        self.bbox = convert_x_to_bbox(self.kf.x)
        self.time_since_update = 0
        self.hits += 1
        self.is_target = is_target_flag
        
        alpha = 0.1
        if self.semantic_feature is not None and sem_feat is not None:
             self.semantic_feature = (1-alpha) * self.semantic_feature + alpha * sem_feat
        if color_feat is not None:
             self.color_hsv = color_feat
             
        # --- NEW: Append the new box to history ---
        self.bbox_history.append({
            "frame": frame_number,
            "x1": int(self.bbox[0]),
            "y1": int(self.bbox[1]),
            "x2": int(self.bbox[2]),
            "y2": int(self.bbox[3])
        })
        # --- END NEW ---

    def is_active(self):
        return self.hits >= self.min_hits

class ObjectTracker:
    def __init__(self, config):
        self.max_age = config['max_age']
        self.min_hits = config['min_hits']
        self.iou_threshold = config['iou_threshold']
        self.weights = config['cost_weights']
        self.tracks = []
        Track._next_id = 1

    def _build_cost_matrix(self, detections, sem_feats, color_feats):
        # ... (This function is UNCHANGED) ...
        num_tracks = len(self.tracks); num_dets = len(detections)
        cost_matrix = np.full((num_tracks, num_dets), 1e5)
        for i, track in enumerate(self.tracks):
            for j in range(num_dets):
                w_mot = self.weights['motion']; w_sem = self.weights['semantic']; w_col = self.weights['color']
                motion_cost = iou_cost(track.bbox, detections[j].bbox)
                if sem_feats[j] is None or color_feats[j] is None:
                    final_cost = motion_cost
                else:
                    sem_cost = (1.0 - np.dot(track.semantic_feature, sem_feats[j])) / 2.0
                    color_cost = calculate_color_distance(track.color_hsv, color_feats[j])
                    final_cost = (w_mot * motion_cost + w_sem * sem_cost + w_col * color_cost)
                cost_matrix[i, j] = final_cost
        return cost_matrix

    def update(self, detections, sem_feats, color_feats, is_target_flags, frame_number): # MODIFIED
        # 1. Predict
        for track in self.tracks: track.predict()

        # 2. Build cost matrix
        cost_matrix = self._build_cost_matrix(detections, sem_feats, color_feats)

        # 3. Assign
        track_indices, det_indices = linear_sum_assignment(cost_matrix)

        # 4. Update matched tracks
        for trk_idx, det_idx in zip(track_indices, det_indices):
            if cost_matrix[trk_idx, det_idx] > (1.0 - self.iou_threshold):
                continue
            self.tracks[trk_idx].update( # MODIFIED
                detections[det_idx], 
                sem_feats[det_idx], 
                color_feats[det_idx],
                is_target_flags[det_idx],
                frame_number # Pass frame number
            )

        # 5. Create new tracks
        unmatched_det_indices = set(range(len(detections))) - set(det_indices)
        for det_idx in unmatched_det_indices:
            new_track = Track( # MODIFIED
                detections[det_idx], 
                sem_feats[det_idx], 
                color_feats[det_idx],
                is_target_flags[det_idx],
                self.min_hits,
                frame_number # Pass frame number
            )
            self.tracks.append(new_track)

        # 6. Prune old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        return [t for t in self.tracks if t.is_active()]