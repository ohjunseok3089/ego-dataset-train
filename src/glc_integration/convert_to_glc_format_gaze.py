#!/usr/bin/env python3
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

class DataConverter:
    def __init__(self, egocom_root, output_root):
        self.egocom_root = Path(egocom_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(exist_ok=True, parents=True)
        
        (self.output_root / "gaze").mkdir(exist_ok=True)
        
    def convert_movement_to_normalized(self, movement_data, roi_width=1280, roi_height=720):
        # Handle null/None movement data (e.g., first frame)
        if not movement_data or movement_data is None:
            return [0.0, 0.0]
            
        focal_length = max(roi_width, roi_height)
        
        h_rad = movement_data.get("horizontal", {}).get("radians", 0.0) if movement_data.get("horizontal") else 0.0
        v_rad = movement_data.get("vertical", {}).get("radians", 0.0) if movement_data.get("vertical") else 0.0
        
        pixel_x = focal_length * np.tan(h_rad)
        pixel_y = focal_length * np.tan(v_rad)
        
        norm_x = pixel_x / roi_width
        norm_y = pixel_y / roi_height
        
        return [norm_x, norm_y]
    
    def calculate_gaze_type(self, current_pos, previous_pos, frame_idx):
        """
        Calculate gaze type based on movement between frames.
        
        Returns:
        0: Fixation (movement ≤ 40 pixels)
        1: Saccade (movement > 40 pixels)
        2: Out-of-bounds/trimmed coordinates
        3: Untracked frames
        """
        if frame_idx == 0:
            return 0  # First frame is always fixation
            
        # Calculate pixel movement
        movement = np.sqrt(
            ((current_pos[0] - previous_pos[0]) * 1280) ** 2 + 
            ((current_pos[1] - previous_pos[1]) * 720) ** 2
        )
        
        # Determine gaze type based on movement threshold
        gaze_type = 0 if movement <= 40 else 1
        
        # Check if coordinates are out of bounds
        if not (0 <= current_pos[0] <= 1 and 0 <= current_pos[1] <= 1):
            gaze_type = 2
            
        return gaze_type
    
    def process_single_video(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        frames = data.get("frames", [])
        metadata = data.get("metadata", {})
        # Handle both old format (video_name) and new format (group_id)
        video_name = metadata.get("video_name") or metadata.get("group_id", "")
        
        gaze_data = []  # Single list for frame, x, y, gaze_type format
        current_pos = [0.5, 0.5]  # Start at center
        previous_pos = [0.5, 0.5]
        
        for i, frame_data in enumerate(frames):
            frame_idx = frame_data.get("frame_index", i)
            
            # Calculate gaze type before updating position
            gaze_type = self.calculate_gaze_type(current_pos, previous_pos, frame_idx)
            
            # Apply movement if available
            next_movement = frame_data.get("next_movement", {})
            if next_movement:
                # Use roi dimensions instead of frame_size_full
                roi = data.get("roi", {}) or metadata.get("roi", {})
                roi_width = roi.get("w", 1280)
                roi_height = roi.get("h", 720)
                
                movement_delta = self.convert_movement_to_normalized(
                    next_movement, 
                    roi_width,
                    roi_height
                )
                
                previous_pos = current_pos.copy()
                current_pos[0] += movement_delta[0]
                current_pos[1] += movement_delta[1]
                
                # Recalculate gaze type after movement
                gaze_type = self.calculate_gaze_type(current_pos, previous_pos, frame_idx)
                
                # Check bounds and update gaze type if needed
                if not (0 <= current_pos[0] <= 1 and 0 <= current_pos[1] <= 1):
                    gaze_type = 2
                    current_pos[0] = np.clip(current_pos[0], 0.0, 1.0)
                    current_pos[1] = np.clip(current_pos[1], 0.0, 1.0)
                
            # Append data in the required format: [frame, x, y, gaze_type]
            gaze_data.append([frame_idx, current_pos[0], current_pos[1], gaze_type])
        
        return gaze_data, video_name
    
    def save_glc_format(self, gaze_data, video_name):
        # Create DataFrame with the required columns: frame, x, y, gaze_type
        gaze_df = pd.DataFrame(gaze_data, columns=['frame', 'x', 'y', 'gaze_type'])
        
        # Save to gaze_frame_label directory with the correct naming format
        label_csv_path = self.output_root / "gaze_frame_label" / f"{video_name}_frame_label.csv"
        gaze_df.to_csv(label_csv_path, index=False)
        
        # Conversion complete silently
        return label_csv_path
    
    def convert_all(self):
        json_dir = self.egocom_root / "joined_ground_truth"
        
        converted_count = 0
        total_frames = 0
        gaze_type_stats = {0: 0, 1: 0, 2: 0, 3: 0}  # Track gaze type distribution
        
        for json_file in json_dir.glob("*.json"):
            try:
                gaze_data, video_name = self.process_single_video(json_file)
                self.save_glc_format(gaze_data, video_name)
                converted_count += 1
                total_frames += len(gaze_data)
                
                # Update gaze type statistics
                for frame_data in gaze_data:
                    gaze_type_stats[frame_data[3]] += 1
                    
            except Exception as e:
                continue
                
        print(f"Converted {converted_count} files ({total_frames} frames) -> {self.output_root}")

if __name__ == "__main__":
    egocom_root = "/mas/robots/prg-egocom/EGOCOM"
    output_root = "/mas/robots/prg-egocom/glc_dataset"
    
    converter = DataConverter(egocom_root, output_root)
    converter.convert_all()