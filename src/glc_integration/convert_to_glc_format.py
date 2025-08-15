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
        
        (self.output_root / "clips.gaze").mkdir(exist_ok=True)
        (self.output_root / "gaze_frame_label").mkdir(exist_ok=True)
        
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
    
    def process_single_video(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        frames = data.get("frames", [])
        metadata = data.get("metadata", {})
        # Handle both old format (video_name) and new format (group_id)
        video_name = metadata.get("video_name") or metadata.get("group_id", "")
        
        gaze_trajectory = []
        
        frame_labels = []
        
        for i, frame_data in enumerate(frames):
            current_pos = [0.5, 0.5]
            frame_idx = frame_data.get("frame_index", i)
            timestamp = frame_data.get("timestamp", i / 30.0)
            
            gaze_trajectory.append({
                "frame": frame_idx,
                "timestamp": timestamp,
                "norm_pos_x": current_pos[0],
                "norm_pos_y": current_pos[1]
            })
            
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
                
                current_pos[0] += movement_delta[0]
                current_pos[1] += movement_delta[1]
                
                current_pos[0] = np.clip(current_pos[0], 0.0, 1.0)
                current_pos[1] = np.clip(current_pos[1], 0.0, 1.0)
                
            frame_labels.append({
                "frame": frame_idx,
                "timestamp": timestamp,
                "social_category": frame_data.get("social_category", "unknown"),
                "speaker_id": frame_data.get("speaker_id", 0)
            })
        
        return gaze_trajectory, frame_labels, video_name
    
    def save_glc_format(self, gaze_trajectory, frame_labels, video_name):
        gaze_df = pd.DataFrame(gaze_trajectory)
        gaze_csv_path = self.output_root / "clips.gaze" / f"{video_name}.csv"
        gaze_df.to_csv(gaze_csv_path, index=False)
        
        label_df = pd.DataFrame(frame_labels)
        label_csv_path = self.output_root / "gaze_frame_label" / f"{video_name}.csv"
        label_df.to_csv(label_csv_path, index=False)
        
        print(f"Conversion complete: {video_name}")
        return gaze_csv_path, label_csv_path
    
    def convert_all(self):
        json_dir = self.egocom_root / "joined_ground_truth"
        
        converted_count = 0
        for json_file in json_dir.glob("*.json"):
            try:
                gaze_trajectory, frame_labels, video_name = self.process_single_video(json_file)
                self.save_glc_format(gaze_trajectory, frame_labels, video_name)
                converted_count += 1
            except Exception as e:
                print(f"Error occurred {json_file}: {e}")
                continue
                
        print(f"\nTotal {converted_count} files converted!")
        print(f"Output directory: {self.output_root}")

if __name__ == "__main__":
    egocom_root = "/mas/robots/prg-egocom/EGOCOM"
    output_root = "./glc_dataset"
    
    converter = DataConverter(egocom_root, output_root)
    converter.convert_all()