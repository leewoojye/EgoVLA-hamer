import os
import glob
from pathlib import Path
from preprocess_pipeline import EgoVLAPipeline
import cv2
import torch
import numpy as np
import sys

# Ensure hamer is in path just in case
sys.path.append(os.getcwd())

def main():
    input_root = "demonstrations"
    output_folder = "demonstrations_preprocessed_npz"
    os.makedirs(output_folder, exist_ok=True)

    # Find videos recursively
    video_files = []
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(('.mov', '.mp4', '.avi')):
                video_files.append(os.path.join(root, file))

    print(f"Found {len(video_files)} videos in {input_root}")

    for i, video_path in enumerate(video_files):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_npz_path = os.path.join(output_folder, f"{video_name}.npz")
        
        print(f"[{i+1}/{len(video_files)}] Processing {video_name}...")
        
        # Check if already done? 
        if os.path.exists(output_npz_path):
            print(f"  Already exists: {output_npz_path}, skipping.")
            continue

        # Create pipeline with saving disabled
        # output_root is required but won't be used for much since we disabled frames/visuals 
        # and override npz path
        pipeline = EgoVLAPipeline(
            video_path=video_path,
            output_root="temp_output_ignore", 
            save_frames=False,
            save_visuals=False
        )
        
        try:
            # Run manually to pass custom output path
            frame_data = pipeline.process_frames()
            if not frame_data:
                print(f"  No frames extracted for {video_name}")
                continue
                
            pipeline.estimate_hand_pose(frame_data, output_npz_path=output_npz_path)
            
        except Exception as e:
            print(f"  Error processing {video_name}: {e}")
            import traceback
            traceback.print_exc()

    print("Batch processing completed.")

if __name__ == "__main__":
    main()
