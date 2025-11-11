import cv2
import numpy as np
from dataclasses import dataclass
from PIL import Image

@dataclass
class Detection:
    bbox: np.ndarray       # [x1, y1, x2, y2]
    mask: np.ndarray      # 2D binary mask
    confidence: float     # Confidence score
    crop: Image.Image = None  # Optional cropped PIL image

def get_dominant_color(image, mask):
    try:
        masked_frame = cv2.bitwise_and(image, image, mask=mask)
        hsv_object = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
        pixels = hsv_object[mask > 0]
        if len(pixels) == 0:
            return  None
        hue_channel = pixels[:, 0]
        hist = cv2.calcHist([hue_channel], [0], None, [180], [0, 180])
        dominant_hue = np.argmax(hist)

        avg_saturation = np.mean(pixels[hue_channel == dominant_hue, 1])
        avg_value = np.mean(pixels[hue_channel == dominant_hue, 2])
        return (dominant_hue, avg_saturation, avg_value)

    except Exception as e:
        print(f"Color detection error: {e}")
        return None
    
def color_distance(hsv1, hsv2):
    if hsv1 is None or hsv2 is None: return 1.0
    h1, s1, v1 = hsv1; h2, s2, v2 = hsv2
    h_dist = min(abs(h1 - h2), 180 - abs(h1 - h2))
    s_dist = abs(s1 - s2); v_dist = abs(v1 - v2)
    norm_h = h_dist / 90.0; norm_s = s_dist / 255.0; norm_v = v_dist / 255.0
    return (0.5 * norm_h) + (0.25 * norm_s) + (0.25 * norm_v)

def draw_tracks(frame: np.ndarray, tracks: list) -> np.ndarray:
    """Draws bounding boxes and labels for all active tracks."""
    for track in tracks:
        if not track.is_active():
            continue
        bbox = track.bbox.astype(int); x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if track.is_target else (255, 100, 100)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID: {track.track_id}"
        if track.is_target: label = f"*** TARGET *** (ID: {track.track_id})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame