import os
import subprocess

class DataConverter:

    def __init__(self, egocom_root, output_root):
        self.input_dir = os.path.join(egocom_root, "parts")
        self.output_dir = os.path.join(output_root, "full_scale.gaze")
        
        if not os.path.isdir(self.input_dir):
            raise FileNotFoundError(f"Input path not found: {self.input_dir}")

    def convert_all(self):
        print(f"Creating output directory: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        videos = [f for f in os.listdir(self.input_dir) if f.endswith('.mp4')]
        
        if not videos:
            print(f"No .mp4 files found in {self.input_dir}")
            return

        print(f"Converting {len(videos)} videos...")
        
        for video_name in videos:
            input_path = os.path.join(self.input_dir, video_name)
            output_path = os.path.join(self.output_dir, video_name)
            
            print(f"[{video_name}] Converting...")
            
            command = [
                'ffmpeg',
                '-i', input_path,
                '-vf', 'scale=-1:256,crop=256:256',
                '-y',
                output_path
            ]
            
            try:
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"[{video_name}] Converted -> {output_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error converting {video_name}: {e.stderr.decode()}")
            except FileNotFoundError:
                print("Error: ffmpeg is not installed or not in PATH")
                print("Please install ffmpeg and try again.")
                return
        
        print("\nAll videos converted successfully.")


if __name__ == "__main__":
    egocom_root = "/mas/robots/prg-egocom/EGOCOM"
    output_root = "/mas/robots/prg-egocom/glc_dataset"
    
    converter = DataConverter(egocom_root, output_root)
    converter.convert_all()