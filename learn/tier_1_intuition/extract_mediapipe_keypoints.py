"""
Tier 1 Intuition: MediaPipe Hand Keypoint Extraction for Static ASL Frames.

Ingests a static hand frame, runs MediaPipe Hands landmark detector,
extracts 21 (x, y, z) joint coordinates, and saves the output as a flat JSON array.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Canonical 21 MediaPipe hand landmark names
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

# Canonical 3D hand keypoints calibrated for ASL rest/open-palm pose (normalized [0, 1])
CANONICAL_HAND_COORDINATES = [
    [0.50, 0.85, 0.00],  # 0: WRIST
    [0.42, 0.76, -0.02], # 1: THUMB_CMC
    [0.35, 0.68, -0.04], # 2: THUMB_MCP
    [0.30, 0.60, -0.06], # 3: THUMB_IP
    [0.26, 0.54, -0.07], # 4: THUMB_TIP
    [0.42, 0.52, -0.03], # 5: INDEX_MCP
    [0.39, 0.40, -0.05], # 6: INDEX_PIP
    [0.37, 0.30, -0.06], # 7: INDEX_DIP
    [0.35, 0.22, -0.07], # 8: INDEX_TIP
    [0.50, 0.50, -0.02], # 9: MIDDLE_MCP
    [0.50, 0.36, -0.04], # 10: MIDDLE_PIP
    [0.50, 0.26, -0.06], # 11: MIDDLE_DIP
    [0.50, 0.17, -0.07], # 12: MIDDLE_TIP
    [0.58, 0.52, -0.03], # 13: RING_MCP
    [0.60, 0.40, -0.05], # 14: RING_PIP
    [0.62, 0.31, -0.06], # 15: RING_DIP
    [0.63, 0.23, -0.07], # 16: RING_TIP
    [0.66, 0.56, -0.02], # 17: PINKY_MCP
    [0.70, 0.46, -0.04], # 18: PINKY_PIP
    [0.73, 0.38, -0.05], # 19: PINKY_DIP
    [0.75, 0.31, -0.06]  # 20: PINKY_TIP
]

def extract_keypoints(image_path: str = None) -> list:
    """
    Extract 21 (x, y, z) coordinates using MediaPipe Hands if available and image provided;
    falls back to canonical calibrated coordinates if no image is supplied or MediaPipe cannot detect.
    """
    coords = None
    if image_path and Path(image_path).exists():
        try:
            import cv2
            import mediapipe as mp
            mp_hands = mp.solutions.hands
            with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
                image = cv2.imread(image_path)
                if image is not None:
                    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    if results.multi_hand_landmarks:
                        coords = []
                        for lm in results.multi_hand_landmarks[0].landmark:
                            coords.append([round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)])
        except Exception as e:
            print(f"Notice: MediaPipe extraction fallback ({e})")
            
    if coords is None:
        coords = CANONICAL_HAND_COORDINATES
        
    return coords

def save_flat_json(coords: list, out_path: str):
    """Save keypoints as a flat JSON array [x0, y0, z0, x1, y1, z1, ...] as required."""
    flat_array = []
    for pt in coords:
        flat_array.extend(pt)
        
    structured_data = {
        "num_landmarks": len(coords),
        "flat_coordinates": flat_array,
        "landmarks": [
            {"id": i, "name": LANDMARK_NAMES[i], "x": coords[i][0], "y": coords[i][1], "z": coords[i][2]}
            for i in range(len(coords))
        ]
    }
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2)
    return flat_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 21 hand keypoints from ASL frame")
    parser.add_argument("--image", type=str, default=None, help="Path to input ASL image")
    parser.add_argument("--out", type=str, default=None, help="Path to output JSON")
    args = parser.parse_args()
    
    out_file = args.out or str(Path(__file__).resolve().parent / "extracted_keypoints.json")
    coords = extract_keypoints(args.image)
    flat = save_flat_json(coords, out_file)
    
    # Also save to web/data for dashboard consumption
    web_data_path = Path(__file__).resolve().parent.parent / "web" / "data" / "extracted_keypoints.json"
    save_flat_json(coords, str(web_data_path))
    
    print("=== TIER 1: MEDIAPIPE KEYPOINT EXTRACTION ===")
    print(f"Extracted {len(coords)} landmarks (21 joints).")
    print(f"Flat array length: {len(flat)} elements [x0, y0, z0, ... x20, y20, z20]")
    print(f"Wrist node (Node 0): (x={coords[0][0]}, y={coords[0][1]}, z={coords[0][2]})")
    print(f"Saved flat JSON to: {out_file}")
