import sys
import torch
import torch.nn as nn
import os
from pathlib import Path
import cv2
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import EgoComDataset

glc_path = os.path.expanduser("~/junseok/GLC")
sys.path.append(str(glc_path))

try:
    from slowfast.models.custom_video_model_builder import *
    import slowfast.utils.checkpoint as cu
    from slowfast.config.defaults import get_cfg
    from slowfast.models import build_model
except ImportError as e:
    print(f"GLC module import failed: {e}")
    print(f"GLC path: {glc_path}")
    sys.exit(1)

class GLC_Baseline:
    def __init__(self, data_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir
        print(f"Device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f} GB")
        
    def create_glc_config(self):
        cfg = get_cfg()
        cfg.MODEL.MODEL_NAME = "GLC_Gaze"
        cfg.MODEL.ARCH = "mvit"
        cfg.MODEL.NUM_CLASSES = 2  # gaze x and y
        
        cfg.DATA.NUM_FRAMES = 8
        cfg.DATA.SAMPLING_RATE = 8
        cfg.DATA.TRAIN_CROP_SIZE = 256
        cfg.DATA.TRAIN_JITTER_SCALES = [256, 320]
        cfg.DATA.TEST_CROP_SIZE = 256
        cfg.DATA.INPUT_CHANNEL_NUM = [3]
        cfg.DATA.TARGET_FPS = 30
        cfg.DATA.USE_OFFSET_SAMPLING = False
        cfg.DATA.GAUSSIAN_KERNEL = 19
        
        # MVIT Configuration - Required for GLC_Gaze model
        cfg.MVIT.ZERO_DECAY_POS_CLS = False
        cfg.MVIT.SEP_POS_EMBED = True
        cfg.MVIT.DEPTH = 16
        cfg.MVIT.NUM_HEADS = 1
        cfg.MVIT.EMBED_DIM = 96
        cfg.MVIT.PATCH_KERNEL = (3, 7, 7)
        cfg.MVIT.PATCH_STRIDE = (2, 4, 4)
        cfg.MVIT.PATCH_PADDING = (1, 3, 3)
        cfg.MVIT.MLP_RATIO = 4.0
        cfg.MVIT.QKV_BIAS = True
        cfg.MVIT.DROPPATH_RATE = 0.2
        cfg.MVIT.NORM = "layernorm"
        cfg.MVIT.MODE = "conv"
        cfg.MVIT.CLS_EMBED_ON = False
        cfg.MVIT.GLOBAL_EMBED_ON = True
        cfg.MVIT.DIM_MUL = [[1, 2.0], [3, 2.0], [14, 2.0]]
        cfg.MVIT.HEAD_MUL = [[1, 2.0], [3, 2.0], [14, 2.0]]
        cfg.MVIT.POOL_KVQ_KERNEL = [3, 3, 3]
        cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE = [1, 8, 8]
        cfg.MVIT.POOL_Q_STRIDE = [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]]
        cfg.MVIT.DROPOUT_RATE = 0.0
        
        cfg.TRAIN.ENABLE = True
        cfg.TRAIN.LR = 1e-4
        cfg.TRAIN.WEIGHT_DECAY = 1e-5
        cfg.TRAIN.BATCH_SIZE = 2
        
        cfg.NUM_GPUS = 1 if torch.cuda.is_available() else 0
        cfg.NUM_SHARDS = 1
        cfg.RNG_SEED = 42
        
        self.cfg = cfg
        return cfg
    

    def load_data(self):
        target_files = [
            "vid_001__day_1__con_1__person_1_part1(0_1920_social_interaction)",
            "vid_001__day_1__con_1__person_1_part1(1980_2370_social_interaction)"
        ]
        
        available_files = []
        json_dir = Path(self.data_dir) / "joined_ground_truth"
        
        for file_name in target_files:
            json_path = json_dir / f"{file_name}.json"
            if json_path.exists():
                available_files.append(file_name)
            else:
                print(f"File not found: {file_name}")
                
        if not available_files:
            print("❌ No available files")
            return None
            
        print(f"✅ {len(available_files)} files found: {available_files}")
        dataset = EgoComDataset(self.data_dir, file_names=available_files)
        
        return dataset
    
    def convert_sample_to_glc_format(self, sample, use_crop=True):
        video_path = sample['video_path']
        target_size = 256
        
        if use_crop:
            original_path = Path(video_path)
            video_name = original_path.name
            cropped_path = Path("/mas/robots/prg-egocom/glc/full_scale.gaze") / video_name
            
            if cropped_path.exists():
                video_path = str(cropped_path)
                print(f"Cropped version used: {video_name}")
            else:
                print(f"Cropped version not found, using original: {video_name}")
                
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Video open failed: {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {width}x{height}, {total_frames} frames, {fps:.1f}fps")
        
        frames = []
        target_frames = self.cfg.DATA.NUM_FRAMES
        sampling_rate = self.cfg.DATA.SAMPLING_RATE
        
        if total_frames > 0:
            if total_frames >= target_frames * sampling_rate:
                indices = list(range(0, target_frames * sampling_rate, sampling_rate))
                print(f"GLC_Gaze standard sampling: {target_frames} frames (indices: {indices})")
            else:
                indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
                print(f"Uniform sampling: {target_frames} frames (total {total_frames} frames)")
                
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if use_crop:
                        frame = cv2.resize(frame, (target_size, target_size))
                    else:
                        h, w = frame.shape[:2]
                        if h != w:
                            size = min(h, w)
                            start_x = (w - size) // 2
                            start_y = (h - size) // 2
                            frame = frame[start_y:start_y+size, start_x:start_x+size]
                        frame = cv2.resize(frame, (target_size, target_size))
                    frames.append(frame)
                    
        cap.release()
        
        if len(frames) < target_frames:
            print(f"Frame shortage: {len(frames)}/{target_frames}, padding...")
            while len(frames) < target_frames:
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((target_size, target_size, 3), dtype=np.uint8))
        
        frames = frames[:target_frames]
        video_tensor = np.array(frames)  # (T, H, W, C)
        video_tensor = torch.from_numpy(video_tensor).float()
        video_tensor = video_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
        video_tensor = video_tensor / 255.0  # normalize pixels [0, 1]
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        video_tensor = (video_tensor - mean) / std
        
        video_tensor = video_tensor.unsqueeze(0)
        
        print(f"✅ Video conversion completed: {video_tensor.shape}")
        print(f"  GLC_Gaze shape: (B=1, C=3, T={target_frames}, H={target_size}, W={target_size})")
        
        return video_tensor
    
    def test_model_with_real_data(self, use_cropped=True):
        print(f"\n🚀 GLC_Gaze real data test started")
        
        # Cropped 디렉토리 확인
        cropped_dir = Path("/mas/robots/prg-egocom/glc/full_scale.gaze")
        if use_cropped and not cropped_dir.exists():
            print(f"Cropped directory not found: {cropped_dir}")
            use_cropped = False
        elif use_cropped:
            cropped_files = list(cropped_dir.glob("*.MP4"))
            print(f"Cropped videos found: {len(cropped_files)}")
        
        # 데이터 로드
        dataset = self.load_data()  # load_real_dataset -> load_data
        if dataset is None:
            return False
            
        print(f"✅ Dataset loaded: {len(dataset)} samples")
            
        print("\n=== GLC_Gaze model loaded ===")
        try:
            cfg = self.create_glc_config()
            model = build_model(cfg)
            model = model.to(self.device)
            model.eval()
            
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✅ GLC_Gaze model loaded: {total_params:,} parameters")
            
        except Exception as e:
            print(f"❌ GLC_Gaze model load failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n=== Real data inference test ===")
        success_count = 0
        test_samples = min(3, len(dataset))
        
        for i in range(test_samples):
            try:
                print(f"\n--- Sample {i+1}/{test_samples} test ---")
                sample = dataset[i]
                
                print(f"Video path: {sample['video_path']}")
                print(f"Frame index: {sample['num_frames']}")
                # print(f"Head movement: {sample['head_movements']}")
                # print(f"Next movement: {sample['next_movements']}")
                # print(f"Social category: {sample['social_categories']}")
                
                video_tensor = self.convert_sample_to_glc_format(sample, use_crop=use_cropped)
                if video_tensor is None:
                    print(f"❌ Sample {i+1} video conversion failed")
                    continue
                
                video_tensor = video_tensor.to(self.device)
                
                print(f"🔍 Tensor debugging:")
                print(f"  video_tensor.shape: {video_tensor.shape}")
                print(f"  video_tensor.dim(): {video_tensor.dim()}")
                print(f"  Expected: (B, C, T, H, W) = (1, 3, 8, 256, 256)")
                
                if video_tensor.dim() == 4:
                    print("  ⚠️ Batch dimension missing - adding...")
                    video_tensor = video_tensor.unsqueeze(0)
                    print(f"  ✅ Modified: {video_tensor.shape}")
                
                with torch.no_grad():
                    output = model(video_tensor)
                
                print(f"✅ GLC_Gaze inference successful!")
                print(f"  Input: {video_tensor.shape}")
                print(f"  Output: {output.shape}")
                print(f"  Output type: {type(output)}")
                
                if isinstance(output, torch.Tensor):
                    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
                    print(f"  Output sample: {output.flatten()[:5].cpu().numpy()}")
                
                success_count += 1
                
            except Exception as e:
                print(f"❌ Sample {i+1} test failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n=== Result ===")
        print(f"Success samples: {success_count}/{test_samples}")
        
        return success_count > 0
    
    def test_training_compatibility(self):
        print("\n=== GLC_Gaze training compatibility test ===")
        
        try:
            dataset = self.load_data()
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
            print(f"Model output shape: {output.shape}")
            
            if len(output.shape) == 3:
                if real_target.size(1) != output.size(1):
                    print(f"⚠️ Frame number mismatch: target={real_target.size(1)}, output={output.size(1)}")
                    if real_target.size(1) > output.size(1):
                        real_target = real_target[:, :output.size(1), :]
                    else:
                        last_movement = real_target[:, -1:, :]
                        padding_needed = output.size(1) - real_target.size(1)
                        padding = last_movement.repeat(1, padding_needed, 1)
                        real_target = torch.cat([real_target, padding], dim=1)
                    print(f"✅ Adjusted target shape: {real_target.shape}")
                        
            elif len(output.shape) == 2:
                real_target = real_target.mean(dim=1)  # (B, 2) - 평균 movement
                print(f"✅ Single prediction target: {real_target.shape}")
            
            else:
                print(f"⚠️ Unexpected model output: {output.shape}")
                return False
            
            loss = criterion(output, real_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"✅ GLC_Gaze training compatibility test successful!")
            print(f"  Using real next_movement")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Target movement example: {real_target[0, 0, :].cpu().numpy()} (first frame)")
            print(f"  Predicted movement: {output[0, 0, :].detach().cpu().numpy()} (first frame)")
            print(f"  Gradient exists: {'Yes' if any(p.grad is not None for p in model.parameters()) else 'No'}")
            
            return True
            
        except Exception as e:
            print(f"❌ GLC_Gaze training compatibility test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    data_dir = "/mas/robots/prg-egocom/EGOCOM"
    
    if not Path(data_dir).exists():
        print(f"❌ Data path not found: {data_dir}")
        return
    
    print("🎯 Goal: GLC_Gaze model + our data compatibility check")
    print("=" * 60)
    
    tester = GLC_Baseline(data_dir)
    
    inference_success = tester.test_model_with_real_data(use_cropped=True)
    
    if inference_success:
        print("\nGLC_Gaze inference test successful!")
        
        training_success = tester.test_training_compatibility()
        
        if training_success:
            print("\nAll tests successful!")

        else:
            print("\nInference works but training has issues")
    else:
        print("\n❌ GLC_Gaze inference test failed")
        print("Model or data conversion issues need to be resolved")

if __name__ == "__main__":
    main()