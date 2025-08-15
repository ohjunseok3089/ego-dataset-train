import sys
import torch
import torch.nn as nn
import os
from pathlib import Path
import cv2
import numpy as np
from data_loader import EgoComDataset

glc_path = "~/junseok/GLC"
sys.path.append(str(glc_path))

from slowfast.models.custom_video_model_builder import *
import slowfast.utils.checkpoint as cu
from slowfast.config.defaults import get_cfg
from slowfast.models import build_model

class GLC_Baseline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def create_glc_config(self):
        cfg = get_cfg()
        cfg.MODEL.MODEL_NAME = "GLC_Gaze"
        cfg.MODEL.ARCH = "mvit"
        cfg.MODEL.NUM_CLASSES = 2 # gaze x and y
        
        cfg.DATA.NUM_FRAMES = 8
        cfg.DATA.SAMPLING_RATE = 8
        cfg.DATA.TRAIN_CROP_SIZE = 256
        cfg.DATA.TRAIN_JITTER_SCALES = [256, 320]
        cfg.DATA.TEST_CROP_SIZE = 256
        cfg.DATA.INPUT_CHANNEL_NUM = [3]
        cfg.DATA.TARGET_FPS = 30
        cfg.DATA.USE_OFFSET_SAMPLING = False
        cfg.DATA.GAUSSIAN_KERNEL = 19
        
        cfg.TRAIN.ENABLE = True
        cfg.TRAIN.LR = 1e-4
        cfg.TRAIN.WEIGHT_DECAY = 1e-5
        
        cfg.NUM_GPUS = 1 
        cfg.NUM_SHARDS = 1
        cfg.RNG_SEED = 42
        
        self.cfg = cfg
        return cfg
    
    def load_data(self):
        target_files = [
            "vid_001__day_1__con_1__person_1_part1(0_1920_social_interaction).MP4",
            "vid_001__day_1__con_1__person_1_part1(1980_2370_social_interaction).MP4"
        ]
        
        avaliable_files = []
        json_dir = Path(self.data_dir) / "joined_ground_truth"
        
        for file_name in target_files:
            json_path = json_dir / f"{file_name}.json"
            if json_path.exists():
                avaliable_files.append(file_name)
                
        test_files = avaliable_files
        dataset = EgoComDataset(self.data_dir, test_files)
        
        return dataset
    
    def convert_sample_to_glc_format(self, sample, use_crop=True):
        video_path = sample[video_path]
        target_size = 256
        if use_crop:
            original_path = Path(video_path)
            video_name = original_path.name
            cropped_path = Path("/mas/robots/prg-egocom/glc/full_scale.gaze") / video_name
            
            if cropped_path.exists():
                video_path = str(cropped_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frames = []
        target_frames = self.cfg.DATA.NUM_FRAMES
        sampling_rate = self.cfg.DATA.SAMPLING_RATE
        
        if total_frames > 0:
            if total_frames >= target_frames * sampling_rate:
                indices = list(range(0, target_frames * sampling_rate, sampling_rate))
            else:
                indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
                
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if use_crop:
                        frame = cv2.resize(frame, (target_size, target_size))
                    else:
                        height, width = frame.shape[:2]
                        if height != width:
                            size = min(height, width)
                            start_x = (width - size) // 2
                            start_y = (height - size) // 2
                            frame = frame[start_y:start_y+size, start_x:start_x+size]
                        frame = cv2.resize(frame, (target_size, target_size))
                    frames.append(frame)
        cap.release()
        frames = frames[:target_frames]
        video_tensor = np.array(frames)
        video_tensor = torch.from_numpy(video_tensor).float()
        video_tensor = video_tensor.permute(3, 0, 1, 2)
        video_tensor = video_tensor / 255.0 # normalize pixels
        

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        video_tensor = (video_tensor - mean) / std
        
        video_tensor = video_tensor.unsqueeze(0)
        
        return video_tensor
    
    def test_model_with_real_data(self, use_cropped=True):

        cropped_dir = Path("/mas/robots/prg-egocom/glc/full_scale.gaze")
        if use_cropped and not cropped_dir.exists():
            use_cropped = False
        elif use_cropped:
            cropped_files = list(cropped_dir.glob("*.MP4"))
        
        dataset = self.load_real_dataset(max_samples=3)
        if dataset is None:
            return False
            
        try:
            cfg = self.create_glc_config()
            model = build_model(cfg)
            model = model.to(self.device)
            model.eval()
            
            total_params = sum(p.numel() for p in model.parameters())
            
        except Exception as e:
            return False
        
        success_count = 0
        for i in range(min(3, len(dataset))):
            try:
                sample = dataset[i]
                
                print(f"Video path: {sample['video_path']}")
                print(f"Frame index: {sample['frame_idx']}")
                print(f"Head movement: {sample['head_movement']}")
                print(f"Next movement: {sample['next_movement']}")
                print(f"Social category: {sample['social_category']}")
                
                video_tensor = self.convert_sample_to_glc_format(sample, use_cropped=use_cropped)
                if video_tensor is None:
                    print(f"Sample {i+1} video conversion failed")
                    continue
                
                video_tensor = video_tensor.to(self.device)
                
                with torch.no_grad():
                    output = model(video_tensor)
                
                print(f"Inference successful")
                print(f"  Input: {video_tensor.shape}")
                print(f"  Output: {output.shape}")
                print(f"  Output type: {type(output)}")
                
                if isinstance(output, torch.Tensor):
                    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
                    print(f"  Output sample: {output.flatten()[:5].cpu().numpy()}")
                
                success_count += 1
                
            except Exception as e:
                print(f"Sample {i+1} test failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Result: {success_count}/{min(3, len(dataset))}")
        
        return success_count > 0
    
    def test_training_compatibility(self):
        try:
            dataset = self.load_real_dataset(max_samples=1)
            if dataset is None or len(dataset) == 0:
                return False
            
            cfg = self.create_glc_config()
            model = build_model(cfg)
            model = model.to(self.device)
            model.train()
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.MSELoss()
            
            sample = dataset[0]
            video_tensor = self.convert_sample_to_glc_format(sample)
            if video_tensor is None:
                return False
                
            video_tensor = video_tensor.to(self.device)
            
            batch_size = video_tensor.size(0)
            dummy_target = torch.randn(batch_size, 2).to(self.device)  # (B, 2)
            
            output = model(video_tensor)
            
            if len(output.shape) == 3:  # (B, T, 2)
                dummy_target = dummy_target.unsqueeze(1).expand(-1, output.size(1), -1)
            
            loss = criterion(output, dummy_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Training compatibility test successful")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Gradient exists: {'Yes' if any(p.grad is not None for p in model.parameters()) else 'No'}")
            
            return True
            
        except Exception as e:
            print(f"Training compatibility test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    data_dir = "/mas/robots/prg-egocom/EGOCOM"
    
    if not Path(data_dir).exists():
        return
    
    tester = GLC_Baseline(data_dir)
    
    inference_success = tester.test_model_with_real_data()
    
    if inference_success:
        training_success = tester.test_training_compatibility()
        
        if training_success:
            print("All tests successful")
            print("Model can be trained with our data")
        else:
            print("Inference works but training has issues")
    else:
        print("Inference test failed")
        print("Model or data conversion issues need to be resolved")

if __name__ == "__main__":
    main()