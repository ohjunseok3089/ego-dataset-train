import json
import os
import glob
import re
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

class EgoComDataset(Dataset):
    def __init__(self, data_dir, file_names, max_detections=5,transform=None):
        self.data_dir = data_dir
        self.json_dir = os.path.join(data_dir, "joined_ground_truth")
        self.transform = transform
        self.max_detections = max_detections
        
        print("[EgoComDataset] Loading data from {} ({} files)".format(self.json_dir, len(file_names)))
        print("[EgoComDataset] Using max_detections: {}".format(self.max_detections))
        
        df = pd.read_csv(os.path.join(data_dir, "transcript", "ground_truth_transcriptions_with_frames.csv"))
        
        self.samples = []
        self.cache = {}
        for file_name in file_names:
            json_path = os.path.join(self.json_dir, f"{file_name}.json")
            if not os.path.exists(json_path):
                print(f"[EgoComDataset] JSON file not found: {json_path}")
                continue
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                num_frames = len(data.get("frames", []))
                if num_frames > 0:
                    for i in range(num_frames):
                        self.samples.append((json_path, i))
            except Exception as e:
                print(f"[EgoComDataset] Error loading JSON file: {json_path} - {e}")
                continue
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        json_path, frame_idx = self.samples[idx]
        if json_path not in self.cache:
            with open(json_path, "r") as f:
                self.cache[json_path] = json.load(f)
                
        data = self.cache[json_path]
        
        frame_data = data["frames"][frame_idx]
    
        metadata = data["metadata"]
        
        # Video path (already contains extension in metadata)
        video_path = metadata.get("video_path", "")
        
        # Co-tracker's red circle (not sure to add this)
        
        # Head movement (handle null values for first frame)
        head_movement_data = frame_data.get("head_movement")
        next_movement_data = frame_data.get("next_movement")
        
        if head_movement_data and head_movement_data is not None:
            h_rad = head_movement_data.get("horizontal", {}).get("radians", 0.0) if head_movement_data.get("horizontal") else 0.0
            v_rad = head_movement_data.get("vertical", {}).get("radians", 0.0) if head_movement_data.get("vertical") else 0.0
            head_movement = torch.tensor([h_rad, v_rad], dtype=torch.float32)
        else:
            head_movement = torch.zeros(2, dtype=torch.float32)
            
        if next_movement_data and next_movement_data is not None:
            h_rad = next_movement_data.get("horizontal", {}).get("radians", 0.0) if next_movement_data.get("horizontal") else 0.0
            v_rad = next_movement_data.get("vertical", {}).get("radians", 0.0) if next_movement_data.get("vertical") else 0.0
            next_movement = torch.tensor([h_rad, v_rad], dtype=torch.float32)
        else:
            next_movement = torch.zeros(2, dtype=torch.float32)
        # Body and face detections
        body_detections = frame_data.get("body_detection", []) or []
        face_detections = frame_data.get("face_detection", []) or []
        social_category = frame_data.get("social_category", "unknown")
        speaker_id = frame_data.get("speaker_id", 0)
        
        
        body_boxes = self._process_detections(body_detections)
        face_boxes = self._process_detections(face_detections)
        
        # social_category (per-frame) with fallback from filename
        if not social_category or social_category == "unknown":
            social_category = self._infer_social_category_from_name(metadata)

        # speaker_id (per-frame)
        

        sample = {
            "video_path": video_path,
            "frame_idx": frame_idx,
            "head_movement": head_movement,
            "next_movement": next_movement,
            "body_boxes": body_boxes,
            "face_boxes": face_boxes,
            "social_category": social_category,
            "speaker_id": speaker_id,
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
        

    def _process_detections(self, detections):
        """ We need to flatten the detections in to static dimensions """
        boxes = []
        for det in detections[:self.max_detections]:
            boxes.append([det['x1'], det['y1'], det['x2'], det['y2']])
        
        num_padding = self.max_detections - len(boxes)
        if num_padding > 0:
            boxes.extend([[0, 0, 0, 0]] * num_padding)
            
        boxes = torch.tensor(boxes, dtype=torch.float32)
        return boxes

    def _infer_social_category_from_name(self, metadata):
        """Infer social category from video name/path, e.g.,
        vid_..._part1(0_1920_social_interaction).MP4 -> social_interaction
        """
        # Handle both old format (video_name) and new format (group_id)
        name = metadata.get("video_name") or metadata.get("group_id") or os.path.basename(metadata.get("video_path", ""))
        if not name:
            return "unknown"
        # Extract text inside parentheses
        match = re.search(r"\(([^)]*)\)", name)
        if not match:
            return "unknown"
        inside = match.group(1)
        # Expect pattern like start_end_label (underscored). Take last token as label
        parts = inside.split("_")
        if len(parts) < 3:
            return "unknown"
        label = parts[-1]
        return label or "unknown"
    
    
if __name__ == "__main__":
    # Temporary test
    # Config file is needed for batch_size, learning rate, etc.
    BATCH_SIZE = 2
    SHUFFLE = True
    
    data_dir = "/mas/robots/prg-egocom/EGOCOM"
    target_files = [
        'vid_001__day_1__con_1__person_1_part1(0_1920_social_interaction)',
        'vid_001__day_1__con_1__person_1_part1(1980_2370_social_interaction)'
    ]
    dataset = EgoComDataset(data_dir, file_names=target_files)
    
    if len(dataset) > 0:
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
        print(f"Loaded {len(dataset)} samples")
    
    for batch in data_loader:
        print(batch)
        break