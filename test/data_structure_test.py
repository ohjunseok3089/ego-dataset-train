import torch
from src.data_loader import EgoComDataset

def test_data_structure():
    
    data_dir = "/mas/robots/prg-egocom/EGOCOM"
    target_files = [
        'vid_001__day_1__con_1__person_1_part1(0_1920_social_interaction)',
        'vid_001__day_1__con_1__person_1_part1(1980_2370_social_interaction)'
    ]
    
    dataset = EgoComDataset(data_dir, file_names=target_files)
    
    if len(dataset) > 0:
        sample = dataset[0]
        
        print("=== Data structure check ===")
        print(f"Sample keys: {list(sample.keys())}")
        
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                if key in ['head_movement', 'next_movement']:
                    print(f"  -> values: {value}")
            else:
                print(f"{key}: {type(value)} = {value}")
        
        print("\n=== Head Movement vs Next Movement ===")
        if 'head_movement' in sample and 'next_movement' in sample:
            head_mov = sample['head_movement']
            next_mov = sample['next_movement']
            print(f"Head movement: {head_mov}")
            print(f"Next movement: {next_mov}")
            print(f"Difference: {next_mov - head_mov}")
        
        return sample
    else:
        print("No samples found!")
        return None

if __name__ == "__main__":
    sample = test_data_structure()