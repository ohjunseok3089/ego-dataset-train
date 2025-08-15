#!/usr/bin/env python3
import sys
import os
import time
from pathlib import Path

try:
    from convert_to_glc_format import DataConverter
    from custom_glc_dataset import CustomGLCDataset, test_dataset
    from glc_trainer import GLCTrainer
except ImportError as e:
    print(f"Import error: {e}")
    print("Check if all files are in the same directory")
    sys.exit(1)

def check_environment():
    print("=== Environment check ===")
    
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f} GB")
    
    egocom_path = Path("/mas/robots/prg-egocom/EGOCOM")
    if not egocom_path.exists():
        print(f"EGOCOM path not found: {egocom_path}")
        return False
    else:
        print(f"EGOCOM path found: {egocom_path}")
    
    glc_path = Path("./GLC")
    if not glc_path.exists():
        print(f"GLC repo not found: {glc_path}")
        print("Run git clone https://github.com/BolinLai/GLC.git")
        return False
    else:
        print(f"GLC repo found: {glc_path}")
    
    return True

def run_full_pipeline():
    print("\n=== Full pipeline run ===")
    
    egocom_root = "/mas/robots/prg-egocom/EGOCOM"
    output_root = "./glc_dataset"
    
    try:
        print("\n1️⃣ Data conversion...")
        converter = DataConverter(egocom_root, output_root)
        
        json_dir = Path(egocom_root) / "joined_ground_truth"
        json_files = list(json_dir.glob("*.json"))[:5]
        
        converted_count = 0
        for json_file in json_files:
            try:
                gaze_trajectory, frame_labels, video_name = converter.process_single_video(json_file)
                converter.save_glc_format(gaze_trajectory, frame_labels, video_name)
                converted_count += 1
                print(f"Conversion complete: {video_name}")
            except Exception as e:
                print(f"Conversion failed {json_file}: {e}")
                continue
        
        if converted_count == 0:
            print("Data conversion failed")
            return False
        
        print(f"{converted_count} files converted")
        
        print("\nDataset test...")
        dataset_success = test_dataset()
        
        if not dataset_success:
            print("Dataset test failed")
            return False
        
        print("Dataset test success")
        
        print("\nModel training test...")
        
        try:
            trainer = GLCTrainer()
            
            trainer.train(num_epochs=1)
            
            print("Model training test success")
            
        except Exception as e:
            print(f"Model training error: {e}")
            print("Data conversion and loading are successful, so only the model part needs to be modified")
            return "partial"
        
        return True
        
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_data_only_test():
    print("\n=== Data pipeline only test ===")
    
    egocom_root = "/mas/robots/prg-egocom/EGOCOM"
    output_root = "./glc_dataset_test"
    
    print("1️⃣ Sample data conversion...")
    
    converter = DataConverter(egocom_root, output_root)
    json_dir = Path(egocom_root) / "joined_ground_truth"
    json_files = list(json_dir.glob("*.json"))
    
    if not json_files:
        print("JSON file not found")
        return False
    
    test_file = json_files[0]
    try:
        gaze_trajectory, frame_labels, video_name = converter.process_single_video(test_file)
        converter.save_glc_format(gaze_trajectory, frame_labels, video_name)
        print(f"Conversion success: {video_name}")
        
        print(f"Gaze trajectory sample ({len(gaze_trajectory)} frames):")
        for i in range(min(5, len(gaze_trajectory))):
            g = gaze_trajectory[i]
            print(f"  Frame {i}: x={g['norm_pos_x']:.3f}, y={g['norm_pos_y']:.3f}")
            
    except Exception as e:
        print(f"Conversion failed: {e}")
        return False
    
    print("\nDataset loading test...")
    
    dataset = CustomGLCDataset(
        data_root=output_root,
        video_root=egocom_root,
        num_frames=8,
        sampling_rate=4
    )
    
    if len(dataset) == 0:
        print("Dataset is empty")
        return False
    
    try:
        sample = dataset[0]
        print(f"Data loading success:")
        print(f"  Video: {sample['video'].shape}")
        print(f"  Gaze: {sample['gaze'].shape}")
        print(f"  Name: {sample['video_name']}")
        
        gaze_data = sample['gaze'].numpy()
        print(f"  Gaze range: x=[{gaze_data[:,0].min():.3f}, {gaze_data[:,0].max():.3f}], " +
              f"y=[{gaze_data[:,1].min():.3f}, {gaze_data[:,1].max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 GLC quick test")
    print("=" * 50)
    
    if not check_environment():
        print("\nEnvironment check failed")
        return
    
    print("\nWhat test to run?")
    print("1. Full pipeline (data + model)")
    print("2. Data pipeline only (safe)")
    print("3. Exit")
    
    choice = input("\nSelect (1/2/3): ").strip()
    
    if choice == "1":
        print("\nRunning full pipeline...")
        start_time = time.time()
        result = run_full_pipeline()
        elapsed = time.time() - start_time
        
        print(f"\nExecution time: {elapsed:.1f} seconds")
        
        if result == True:
            print("Full pipeline success!")
            print("Now increase the number of epochs and start actual training.")
        elif result == "partial":
            print("Data part is successful, only the model part needs to be modified")
        else:
            print("Pipeline failed")
            
    elif choice == "2":
        print("\nRunning data pipeline only...")
        start_time = time.time()
        result = run_data_only_test()
        elapsed = time.time() - start_time
        
        print(f"\nExecution time: {elapsed:.1f} seconds")
        
        if result:
            print("Data pipeline success!")
            print("Now integrate the GLC model")
        else:
            print("Data pipeline failed")
    
    elif choice == "3":
        print("Exiting...")
        return
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()